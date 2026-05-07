import os, io, pickle
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import gdown
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input

# ═══════════════════════════════════════════════════════════
# KERAS COMPATIBILITY PATCH
# Render uses Keras 3.x which breaks old .h5 models.
# We replace InputLayer and register DTypePolicy so the
# model config deserializes correctly regardless of version.
# ═══════════════════════════════════════════════════════════

class _CompatInputLayer(tf.keras.layers.Layer):
    """
    Drop-in replacement for InputLayer that silently accepts
    all legacy kwargs (batch_shape, optional, sparse, ragged).
    """
    def __init__(self, shape=None, batch_size=None, dtype=None,
                 sparse=False, ragged=False, name=None,
                 # legacy kwargs — accepted and ignored
                 batch_shape=None, optional=None, **kwargs):
        # batch_shape → shape
        if shape is None and batch_shape is not None:
            shape = batch_shape[1:]  # strip batch dim
        super().__init__(name=name, dtype=dtype, **kwargs)
        self._output_shape = shape
        self._batch_size   = batch_size

    def call(self, inputs):
        return inputs

    @classmethod
    def from_config(cls, config):
        # Accept any config key without crashing
        config.pop("optional", None)
        return cls(**config)


class _DTypePolicy:
    """Stub for keras DTypePolicy — only needs to survive deserialization."""
    def __init__(self, name="float32", **kwargs):
        self.name = name

    def get_config(self):
        return {"name": self.name}


def _apply_keras_patches():
    """Monkey-patch Keras internals once at startup."""
    try:
        # Patch InputLayer
        tf.keras.layers.InputLayer = _CompatInputLayer

        # Register DTypePolicy under all names Keras might look for
        for alias in ("DTypePolicy", "dtype_policy", "Policy"):
            try:
                tf.keras.utils.get_custom_objects()[alias] = _DTypePolicy
            except Exception:
                pass

        # Also patch ZeroPadding2D dtype config issue
        _orig_zp_from_config = tf.keras.layers.ZeroPadding2D.from_config

        @classmethod
        def _zp_from_config(cls, config):
            config.pop("dtype", None)
            config_clean = {k: v for k, v in config.items()
                            if not isinstance(v, dict) or "class_name" not in v}
            try:
                return _orig_zp_from_config.__func__(cls, config)
            except Exception:
                return _orig_zp_from_config.__func__(cls, config_clean)

        tf.keras.layers.ZeroPadding2D.from_config = _zp_from_config

        print("✅ Keras compatibility patches applied.")
    except Exception as e:
        print(f"⚠ Keras patch warning: {e}")


_apply_keras_patches()

# ═══════════════════════════════════════════════════════════

app = FastAPI(title="OcularAI Eye Disease Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GDRIVE_FILES = {
    "densenet121_finetuned.h5": "1FoVidXa3rt5ANpSPNzUrxwCN1Xv9Wg_M",
    "best_clf.pkl":             "1jBE3tYjqG_nWw5Y7KRazD1SvDdHrhHgZ",
    "selector.pkl":             "1vFyw9karH1eGttLdfP-N0aK2uoXHOJpf",
    "scaler.pkl":               "156xcXLRkd4h2CWvBPIjZQ-MYu4RoJS3R",
    "pca.pkl":                  "1fNf7Hp9Yk7yuooqohkiZ5sAmTB_8ut2e",
}

MODEL_DIR = "/tmp/ocularai_models"
os.makedirs(MODEL_DIR, exist_ok=True)

CLASS_NAMES = ["AMD", "CNV", "CSR", "DME", "DR", "DRUSEN", "MH", "NORMAL"]

DISEASE_INFO = {
    "AMD":    {"full": "Age-related Macular Degeneration", "severity": "High",
               "desc": "Progressive degeneration of the macula. Wet AMD needs urgent anti-VEGF injections; dry AMD managed with AREDS2 supplements."},
    "CNV":    {"full": "Choroidal Neovascularization", "severity": "High",
               "desc": "Abnormal blood vessel growth beneath the retina. Requires urgent anti-VEGF therapy (ranibizumab, bevacizumab)."},
    "CSR":    {"full": "Central Serous Retinopathy", "severity": "Moderate",
               "desc": "Subretinal fluid accumulation often stress-related. Most cases resolve in 3 months. Chronic cases need photodynamic therapy."},
    "DME":    {"full": "Diabetic Macular Edema", "severity": "High",
               "desc": "Fluid in the macula due to diabetic vascular leakage. Anti-VEGF or laser treatment required alongside glycemic control."},
    "DR":     {"full": "Diabetic Retinopathy", "severity": "High",
               "desc": "Damage to retinal blood vessels from long-term diabetes. Laser therapy, anti-VEGF, and strict blood sugar control needed."},
    "DRUSEN": {"full": "Drusen Deposits", "severity": "Moderate",
               "desc": "Yellow deposits under the retina — early sign of AMD. Regular monitoring essential. AREDS2 supplements may slow progression."},
    "MH":     {"full": "Macular Hole", "severity": "High",
               "desc": "Full-thickness defect in central retina. Pars plana vitrectomy is highly effective (>90% success) when performed early."},
    "NORMAL": {"full": "Normal Healthy Retina", "severity": "None",
               "desc": "No pathological changes detected. Retinal layers appear intact. Continue routine annual eye examinations."},
}

_models = {}


# ── Download helper ──────────────────────────────────
def download_file(filename: str) -> str:
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        print(f"✅ Cached: {filename}")
        return path
    file_id = GDRIVE_FILES.get(filename)
    if not file_id or "YOUR_" in file_id:
        raise RuntimeError(f"File ID not set for '{filename}'.")
    print(f"⬇ Downloading {filename} ...")
    out = gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)
    if not out or not os.path.exists(path):
        raise RuntimeError(f"Download failed for '{filename}'. Check Drive sharing.")
    print(f"✅ {filename} ({os.path.getsize(path)/1024/1024:.1f} MB)")
    return path


def load_pkl(filename: str):
    try:
        with open(download_file(filename), "rb") as f:
            obj = pickle.load(f)
        print(f"✅ Loaded {filename}: {type(obj).__name__}")
        return obj
    except Exception as e:
        print(f"⚠ Skipping {filename}: {e}")
        return None


# ── Model loading ────────────────────────────────────
def load_all_models():
    if _models.get("ready"):
        return

    print("=" * 55)
    print("🔄 Loading all models...")

    model_path = download_file("densenet121_finetuned.h5")
    full_model = None

    custom_objs = {
        "InputLayer":    _CompatInputLayer,
        "DTypePolicy":   _DTypePolicy,
        "dtype_policy":  _DTypePolicy,
        "Policy":        _DTypePolicy,
        "ZeroPadding2D": tf.keras.layers.ZeroPadding2D,
    }

    # Try 1: load_model with custom_objects (patches already applied globally)
    try:
        full_model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objs,
            compile=False
        )
        print("✅ Model loaded (custom_objects).")
    except Exception as e1:
        print(f"⚠ custom_objects load failed: {e1}")

    # Try 2: h5py — strip ALL problematic keys from config JSON
    if full_model is None:
        try:
            print("Trying deep h5py JSON strip...")
            import h5py, json

            with h5py.File(model_path, "r") as f:
                raw_cfg = f.attrs.get("model_config")

            cfg_str = raw_cfg if isinstance(raw_cfg, str) else raw_cfg.decode("utf-8")
            cfg = json.loads(cfg_str)

            def _clean_config(obj):
                """Recursively clean layer configs."""
                if isinstance(obj, dict):
                    # Fix InputLayer
                    if obj.get("class_name") == "InputLayer":
                        c = obj.get("config", {})
                        shape = c.get("batch_shape", c.get("shape", [None, 224, 224, 3]))
                        obj["config"] = {
                            "batch_shape": shape,
                            "dtype":       c.get("dtype", "float32"),
                            "sparse":      False,
                            "ragged":      False,
                            "name":        c.get("name", "input_layer"),
                        }
                    # Fix DTypePolicy inside dtype field
                    if "dtype" in obj and isinstance(obj["dtype"], dict):
                        dtype_cfg = obj["dtype"]
                        if dtype_cfg.get("class_name") in ("DTypePolicy", "Policy"):
                            inner = dtype_cfg.get("config", {})
                            obj["dtype"] = inner.get("name", "float32")
                    return {k: _clean_config(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [_clean_config(i) for i in obj]
                return obj

            cfg = _clean_config(cfg)
            full_model = tf.keras.models.model_from_json(
                json.dumps(cfg),
                custom_objects=custom_objs
            )
            full_model.load_weights(model_path)
            print("✅ Model loaded (deep h5py strip).")
        except Exception as e2:
            print(f"⚠ Deep h5py strip failed: {e2}")

    # Try 3: normal load (last resort)
    if full_model is None:
        try:
            full_model = tf.keras.models.load_model(model_path, compile=False)
            print("✅ Model loaded (normal).")
        except Exception as e3:
            print(f"⚠ Normal load failed: {e3}")

    if full_model is None:
        raise RuntimeError("All model load attempts failed. Check logs above.")

    print(f"✅ Model ready. Layers: {len(full_model.layers)}")

    # Feature extractor: layers[-3] = Dense(256)
    feat_layer = full_model.layers[-3]
    print(f"✅ Feature layer: {feat_layer.name}")
    _models["extractor"] = tf.keras.Model(
        inputs=full_model.input,
        outputs=feat_layer.output
    )

    # ML components
    _models["clf"]      = load_pkl("best_clf.pkl")
    _models["selector"] = load_pkl("selector.pkl")
    _models["scaler"]   = load_pkl("scaler.pkl")
    _models["pca"]      = load_pkl("pca.pkl")

    if _models["clf"] is None:
        raise RuntimeError("best_clf.pkl failed to load.")

    if _models["selector"] is not None:
        _models["pipeline"] = "kbest"
        print("✅ Pipeline: DenseNet[-3](256) → SelectKBest → SVM")
    elif _models["scaler"] and _models["pca"]:
        _models["pipeline"] = "pca"
        print("✅ Pipeline: DenseNet[-3](256) → Scaler → PCA → SVM")
    else:
        _models["pipeline"] = "direct"
        print("⚠ Pipeline: DenseNet[-3](256) → SVM direct")

    print(f"🚀 Ready! Classes: {CLASS_NAMES}")
    _models["ready"] = True


# ── Lightweight Feature Importance (SHAP-style, no KernelExplainer) ──
def compute_shap(feats_reduced: np.ndarray, pred_class: str) -> list:
    try:
        clf      = _models["clf"]
        cls_idx  = CLASS_NAMES.index(pred_class) if pred_class in CLASS_NAMES else 0
        n_feats  = feats_reduced.shape[1]
        feats_1d = feats_reduced[0]

        # Linear SVM → coef_ directly
        if hasattr(clf, "coef_"):
            coef = clf.coef_
            sv   = coef[cls_idx] * feats_1d if coef.shape[0] > 1 and cls_idx < coef.shape[0] else coef[0] * feats_1d
            abs_sv  = np.abs(sv)
            top_idx = np.argsort(abs_sv)[::-1][:10]
            print("✅ Feature importance (coef_)")
            return [{"feature": f"F{int(i)}", "value": round(float(sv[i]), 4), "abs": round(float(abs_sv[i]), 4)} for i in top_idx]

        # Non-linear SVM → perturbation on top-30
        base_score = clf.decision_function(feats_reduced)
        base_val   = float(base_score[0][cls_idx]) if base_score.ndim > 1 and base_score.shape[1] > cls_idx else float(base_score[0])
        cands      = np.argsort(np.abs(feats_1d))[::-1][:30]
        sv         = np.zeros(n_feats)
        eps        = 1e-2

        for i in cands:
            p        = feats_reduced.copy()
            p[0][i] += eps
            ns       = clf.decision_function(p)
            nv       = float(ns[0][cls_idx]) if ns.ndim > 1 and ns.shape[1] > cls_idx else float(ns[0])
            sv[i]    = (nv - base_val) / eps

        abs_sv  = np.abs(sv)
        top_idx = np.argsort(abs_sv)[::-1][:10]
        print("✅ Feature importance (perturbation)")
        return [{"feature": f"F{int(i)}", "value": round(float(sv[i]), 4), "abs": round(float(abs_sv[i]), 4)} for i in top_idx]

    except Exception as e:
        print(f"⚠ Feature importance error: {e}")
        return []


# ── Prediction ───────────────────────────────────────
def run_prediction(image_bytes: bytes) -> dict:
    load_all_models()

    img   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img   = img.resize((224, 224), Image.LANCZOS)
    arr   = preprocess_input(np.array(img, dtype=np.float32))
    x     = np.expand_dims(arr, axis=0)
    feats = _models["extractor"].predict(x, verbose=0).reshape(1, -1)
    print(f"Features shape: {feats.shape}")

    pipeline = _models["pipeline"]
    if pipeline == "kbest":
        feats = _models["selector"].transform(feats)
    elif pipeline == "pca":
        feats = _models["scaler"].transform(feats)
        feats = _models["pca"].transform(feats)

    clf      = _models["clf"]
    pred_raw = clf.predict(feats)[0]
    pred_class = CLASS_NAMES[int(pred_raw)] if isinstance(pred_raw, (int, np.integer)) and int(pred_raw) < len(CLASS_NAMES) else str(pred_raw)

    if hasattr(clf, "predict_proba"):
        proba_raw = clf.predict_proba(feats)[0]
        if hasattr(clf, "classes_"):
            prob_map = {}
            for c, p in zip(clf.classes_, proba_raw):
                key = CLASS_NAMES[int(c)] if isinstance(c, (int, np.integer)) else str(c)
                prob_map[key] = float(p)
        else:
            prob_map = {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba_raw) if i < len(CLASS_NAMES)}
        sorted_cls = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)
        top1_class, top1_conf = sorted_cls[0]
        top2_class = sorted_cls[1][0] if len(sorted_cls) > 1 else None
        top2_conf  = sorted_cls[1][1] if len(sorted_cls) > 1 else 0.0
    else:
        top1_class, top1_conf = pred_class, 1.0
        top2_class, top2_conf = None, 0.0
        prob_map = {pred_class: 1.0}

    shap_values = compute_shap(feats, top1_class)
    info = DISEASE_INFO.get(top1_class, DISEASE_INFO["NORMAL"])

    return {
        "disease":              top1_class,
        "confidence":           round(top1_conf, 4),
        "full_name":            info["full"],
        "severity":             info["severity"],
        "description":          info["desc"],
        "secondary":            top2_class,
        "secondary_confidence": round(top2_conf, 4),
        "all_probabilities":    {k: round(v, 4) for k, v in prob_map.items()},
        "shap_values":          shap_values,
    }


# ── Routes ───────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "OcularAI Eye Disease Detection"}

@app.get("/health")
def health():
    return {
        "status":       "healthy",
        "model_loaded": _models.get("ready", False),
        "pipeline":     _models.get("pipeline", "not loaded"),
        "classes":      CLASS_NAMES,
    }

@app.get("/debug")
def debug():
    files = {}
    for name in GDRIVE_FILES:
        path = os.path.join(MODEL_DIR, name)
        files[name] = {
            "id_set":     "YOUR_" not in GDRIVE_FILES[name],
            "downloaded": os.path.exists(path),
            "size_mb":    round(os.path.getsize(path)/1024/1024, 2) if os.path.exists(path) else 0,
        }
    return {
        "files":      files,
        "loaded":     _models.get("ready", False),
        "pipeline":   _models.get("pipeline", "not loaded"),
        "class_order":{i: c for i, c in enumerate(CLASS_NAMES)},
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files accepted.")
    contents = await file.read()
    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(400, "Image too large. Max 15 MB.")
    try:
        result = run_prediction(contents)
        return JSONResponse({"success": True, **result})
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(500, f"Prediction failed: {str(e)}")