import os, io, pickle
import shap
import gdown
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import load_model, Model

app = FastAPI(title="OcularAI Eye Disease Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ═══════════════════════════════════════════════════
# Google Drive File IDs
# ═══════════════════════════════════════════════════
GDRIVE_FILES = {
    "densenet121_finetuned.h5": "1FoVidXa3rt5ANpSPNzUrxwCN1Xv9Wg_M",
    "best_clf.pkl":             "1jBE3tYjqG_nWw5Y7KRazD1SvDdHrhHgZ",
    "selector.pkl":             "1vFyw9karH1eGttLdfP-N0aK2uoXHOJpf",
    "scaler.pkl":               "156xcXLRkd4h2CWvBPIjZQ-MYu4RoJS3R",
    "pca.pkl":                  "1fNf7Hp9Yk7yuooqohkiZ5sAmTB_8ut2e",
}

MODEL_DIR = "/tmp/ocularai_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Notebook output থেকে exact class order (alphabetical sorted)
CLASS_NAMES = ["AMD", "CNV", "CSR", "DME", "DR", "DRUSEN", "MH", "NORMAL"]

DISEASE_INFO = {
    "AMD":    {"full": "Age-related Macular Degeneration", "severity": "High",
               "desc": "Progressive degeneration of the macula. Wet AMD needs urgent anti-VEGF injections; dry AMD managed with AREDS2 supplements."},
    "CNV":    {"full": "Choroidal Neovascularization", "severity": "High",
               "desc": "Abnormal blood vessel growth beneath the retina causing rapid central vision loss. Requires urgent anti-VEGF therapy."},
    "CSR":    {"full": "Central Serous Retinopathy", "severity": "Moderate",
               "desc": "Subretinal fluid accumulation often related to stress. Most acute cases resolve in 3 months. Chronic cases may need photodynamic therapy."},
    "DME":    {"full": "Diabetic Macular Edema", "severity": "High",
               "desc": "Fluid in the macula due to diabetic vascular leakage. Leading cause of vision loss in diabetics. Anti-VEGF or laser treatment required."},
    "DR":     {"full": "Diabetic Retinopathy", "severity": "High",
               "desc": "Damage to retinal blood vessels caused by long-term diabetes. Can progress to blindness. Laser therapy, anti-VEGF, and strict blood sugar control needed."},
    "DRUSEN": {"full": "Drusen Deposits", "severity": "Moderate",
               "desc": "Yellow lipid deposits beneath the retina — early sign of AMD. Regular monitoring essential. AREDS2 supplements may slow progression."},
    "MH":     {"full": "Macular Hole", "severity": "High",
               "desc": "Full-thickness defect in the central retina. Pars plana vitrectomy is highly effective (>90% success) when performed early."},
    "NORMAL": {"full": "Normal Healthy Retina", "severity": "None",
               "desc": "No pathological changes detected. Retinal layers appear intact. Continue routine annual eye examinations."},
}

_models = {}
_shap_cache = {}   # background data cache for KernelExplainer


# ═══════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════
def download_file(filename: str) -> str:
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        print(f"✅ Cached: {filename}")
        return path
    file_id = GDRIVE_FILES.get(filename)
    if not file_id or "YOUR_" in file_id:
        raise RuntimeError(f"❌ File ID not set for '{filename}'.")
    print(f"⬇ Downloading {filename} ...")
    out = gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)
    if not out or not os.path.exists(path):
        raise RuntimeError(f"❌ Download failed for '{filename}'. Check sharing settings.")
    print(f"✅ Downloaded {filename} ({os.path.getsize(path)/1024/1024:.1f} MB)")
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


def load_all_models():
    if _models.get("ready"):
        return
    print("=" * 55)
    print("🔄 Loading all models...")

    full_model = load_model(download_file("densenet121_finetuned.h5"))
    print(f"✅ Model loaded. Layers: {len(full_model.layers)}")

    # layers[-3] = Dense(256) → 256-dim features (from notebook cell 21)
    feat_layer = full_model.layers[-3]
    print(f"✅ Feature layer: {feat_layer.name}")
    _models["extractor"] = Model(inputs=full_model.input, outputs=feat_layer.output)

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

    print(f"🚀 Ready! Classes: {CLASS_NAMES}")
    _models["ready"] = True


def compute_shap(feats_reduced: np.ndarray, pred_class: str) -> list:
    """
    SHAP KernelExplainer দিয়ে top-10 feature importance বের করে।
    Notebook cell 35 এর same approach।
    """
    try:
        clf = _models["clf"]
        if not hasattr(clf, "predict_proba"):
            return []

        # Background data: zero vector (fast approximation)
        # Full KernelExplainer এর জন্য training data লাগে
        # এখানে mean=0 background use করছি (fast)
        n_feats = feats_reduced.shape[1]

        if "explainer" not in _shap_cache or _shap_cache.get("n_feats") != n_feats:
            print(f"Building SHAP KernelExplainer (n_feats={n_feats})...")
            bg = np.zeros((1, n_feats))  # single zero background — fast
            _shap_cache["explainer"] = shap.KernelExplainer(
                clf.predict_proba, bg
            )
            _shap_cache["n_feats"] = n_feats
            print("SHAP explainer ready")

        explainer = _shap_cache["explainer"]

        # Calculate SHAP values (nsamples=20 → fast but approximate)
        shap_vals = explainer.shap_values(feats_reduced, nsamples=20)

        # Get class index
        if pred_class in CLASS_NAMES:
            cls_idx = CLASS_NAMES.index(pred_class)
        else:
            cls_idx = 0

        # shap_vals shape: (n_classes, n_samples, n_features) or list
        if isinstance(shap_vals, list):
            sv = shap_vals[cls_idx][0]   # (n_features,)
        else:
            sv = shap_vals[0]

        # Top 10 features by absolute value
        abs_sv   = np.abs(sv)
        top_idx  = np.argsort(abs_sv)[::-1][:10]

        result = []
        for i in top_idx:
            result.append({
                "feature": f"F{int(i)}",
                "value":   round(float(sv[i]), 4),
                "abs":     round(float(abs_sv[i]), 4),
            })

        print(f"SHAP done. Top feature: {result[0] if result else 'none'}")
        return result

    except Exception as e:
        print(f"SHAP error: {e}")
        return []


def run_prediction(image_bytes: bytes) -> dict:
    load_all_models()

    img  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img  = img.resize((224, 224), Image.LANCZOS)
    arr  = preprocess_input(np.array(img, dtype=np.float32))
    x    = np.expand_dims(arr, axis=0)

    feats = _models["extractor"].predict(x, verbose=0).reshape(1, -1)
    print(f"Features: {feats.shape}")

    if _models["pipeline"] == "kbest":
        feats = _models["selector"].transform(feats)
    elif _models["pipeline"] == "pca":
        feats = _models["scaler"].transform(feats)
        feats = _models["pca"].transform(feats)

    clf      = _models["clf"]
    pred_raw = clf.predict(feats)[0]

    if isinstance(pred_raw, (int, np.integer)):
        pred_class = CLASS_NAMES[int(pred_raw)] if int(pred_raw) < len(CLASS_NAMES) else "NORMAL"
    else:
        pred_class = str(pred_raw)

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

    info = DISEASE_INFO.get(top1_class, DISEASE_INFO["NORMAL"])

    # ── SHAP XAI ────────────────────────────────────────
    shap_values = compute_shap(feats, top1_class)

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


# ═══════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════
@app.get("/")
def root():
    return {"status": "ok", "service": "OcularAI — Retinal Eye Disease Detection"}

@app.get("/health")
def health():
    return {
        "status":          "healthy",
        "model_loaded":    _models.get("ready", False),
        "pipeline":        _models.get("pipeline", "not loaded"),
        "classes":         CLASS_NAMES,
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
        "files":         files,
        "models_loaded": _models.get("ready", False),
        "pipeline":      _models.get("pipeline", "not loaded"),
        "class_order":   {i: c for i, c in enumerate(CLASS_NAMES)},
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files accepted.")

    contents = await file.read()
    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(400, "Image too large. Max 15 MB.")

    # ── Direct SVM prediction (no retinal validation) ──
    try:
        result = run_prediction(contents)
        return JSONResponse({"success": True, "is_retinal": True, **result})
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(500, f"Prediction failed: {str(e)}")
