import os, io, pickle, base64, httpx
import gdown
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Keras 3 + old .h5 compat: patch BEFORE any keras/tf import ───────────────
import keras
from keras import layers as _klayers

_orig_base_fc = _klayers.Layer.from_config.__func__
@classmethod
def _compat_layer_fc(cls, config):
    cfg = dict(config)
    if isinstance(cfg.get("dtype"), dict):
        inner = cfg["dtype"].get("config", {})
        cfg["dtype"] = inner.get("name", "float32")
    cfg.pop("batch_shape", None)
    cfg.pop("optional",    None)
    if "input_shape" in cfg and "shape" not in cfg:
        cfg["shape"] = cfg.pop("input_shape")
    return _orig_base_fc(cls, cfg)
_klayers.Layer.from_config = _compat_layer_fc
print("keras Layer.from_config patched")

import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import load_model, Model
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="OcularAI Eye Disease Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GDRIVE_FILES = {
    "densenet121_finetuned.h5": "1FoVidXa3rt5ANpSPNzUrxwCN1Xv9Wg_M",
    "best_clf.pkl":             "1jBE3tYjqG_nWw5Y7KRazD1SvDdHrhHgZ",
    "selector.pkl":             "1vFyw9karH1eGttLdfP-N0aK2uoXHOJpf",
    "scaler.pkl":               "156xcXLRkd4h2CWvBPIjZQ-MYu4RoJS3R",
    "pca.pkl":                  "1fNf7Hp9Yk7yuooqohkiZ5sAmTB_8ut2e",
}

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL_DIR = "/tmp/ocularai_models"
os.makedirs(MODEL_DIR, exist_ok=True)

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

_models     = {}
_shap_cache = {}


async def is_retinal_oct(image_bytes: bytes, media_type: str) -> tuple[bool, str]:
    if not ANTHROPIC_API_KEY:
        return True, "validation_skipped"
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": "claude-sonnet-4-20250514", "max_tokens": 100,
        "system": "You are a medical image classifier. Decide if the image is a retinal OCT scan. Reply ONLY: YES or NO.",
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
            {"type": "text",  "text": "Is this a retinal OCT scan? YES or NO only."}
        ]}]
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json=payload
            )
        answer = resp.json()["content"][0]["text"].strip().upper()
        print(f"Claude validation: {answer}")
        return answer.startswith("YES"), answer
    except Exception as e:
        print(f"Claude validation failed: {e}")
        return True, f"error:{e}"


def download_file(filename: str) -> str:
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        print(f"Cached: {filename}"); return path
    file_id = GDRIVE_FILES.get(filename)
    if not file_id or "YOUR_" in file_id:
        raise RuntimeError(f"File ID not set for '{filename}'.")
    print(f"Downloading {filename} ...")
    out = gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)
    if not out or not os.path.exists(path):
        raise RuntimeError(f"Download failed for '{filename}'.")
    print(f"Downloaded {filename} ({os.path.getsize(path)/1024/1024:.1f} MB)")
    return path


def load_pkl(filename: str):
    try:
        with open(download_file(filename), "rb") as f:
            obj = pickle.load(f)
        print(f"Loaded {filename}: {type(obj).__name__}"); return obj
    except Exception as e:
        print(f"Skipping {filename}: {e}"); return None


def load_all_models():
    if _models.get("ready"): return
    print("=" * 55)
    print("Loading all models...")

    full_model = load_model(download_file("densenet121_finetuned.h5"), compile=False)
    print(f"Model loaded. Layers: {len(full_model.layers)}")

    feat_layer = None
    for layer in reversed(full_model.layers):
        if hasattr(layer, "units") and layer.units == 256:
            feat_layer = layer; break
    if feat_layer is None:
        feat_layer = full_model.layers[-3]
    print(f"Feature layer: {feat_layer.name}")
    _models["extractor"] = Model(inputs=full_model.input, outputs=feat_layer.output)

    _models["clf"]      = load_pkl("best_clf.pkl")
    _models["selector"] = load_pkl("selector.pkl")
    _models["scaler"]   = load_pkl("scaler.pkl")
    _models["pca"]      = load_pkl("pca.pkl")

    if _models["clf"] is None:
        raise RuntimeError("best_clf.pkl failed to load.")

    if _models["selector"] is not None:
        _models["pipeline"] = "kbest"
    elif _models["scaler"] and _models["pca"]:
        _models["pipeline"] = "pca"
    else:
        _models["pipeline"] = "direct"

    print(f"Pipeline: {_models['pipeline']} | Ready!")
    _models["ready"] = True


def compute_shap(feats_reduced: np.ndarray, pred_class: str) -> list:
    """
    SHAP KernelExplainer — memory-safe implementation.
    n_bg = n_feats+50 ensures n_samples > n_features (avoids LassoLarsIC).
    l1_reg='num_features(10)' bypasses LassoLarsIC auto-selection entirely.
    Explainer is cached — only built once per session.
    """
    try:
        import shap as shap_lib

        clf     = _models["clf"]
        kernel  = getattr(clf, "kernel", "rbf")
        n_feats = feats_reduced.shape[1]
        cls_idx = CLASS_NAMES.index(pred_class) if pred_class in CLASS_NAMES else 0
        cache_key = f"{kernel}_{n_feats}"

        if _shap_cache.get("key") != cache_key:
            print(f"Building SHAP explainer: kernel={kernel}, n_feats={n_feats}")
            if kernel == "linear" and hasattr(clf, "coef_"):
                bg = np.zeros((1, n_feats))
                _shap_cache["explainer"] = shap_lib.LinearExplainer(clf, bg)
                _shap_cache["type"] = "linear"
                print("SHAP: LinearExplainer ready")
            else:
                if not hasattr(clf, "predict_proba"):
                    return []
                n_bg = n_feats + 50
                np.random.seed(42)
                bg = np.random.normal(0, 0.1, size=(n_bg, n_feats)).astype(np.float32)
                _shap_cache["explainer"] = shap_lib.KernelExplainer(clf.predict_proba, bg)
                _shap_cache["type"] = "kernel"
                print(f"SHAP: KernelExplainer ready (n_bg={n_bg})")
            _shap_cache["key"] = cache_key

        explainer      = _shap_cache["explainer"]
        explainer_type = _shap_cache["type"]

        if explainer_type == "linear":
            shap_vals = explainer.shap_values(feats_reduced)
        else:
            shap_vals = explainer.shap_values(
                feats_reduced, nsamples=100, l1_reg="num_features(10)", silent=True
            )

        if isinstance(shap_vals, list):
            sv = np.array(shap_vals[cls_idx] if cls_idx < len(shap_vals) else shap_vals[0])
            sv = sv[0] if sv.ndim == 2 else sv
        elif isinstance(shap_vals, np.ndarray):
            sv = shap_vals[cls_idx, 0, :] if shap_vals.ndim == 3 else (shap_vals[0] if shap_vals.ndim == 2 else shap_vals)
        else:
            return []

        abs_sv  = np.abs(sv)
        top_idx = np.argsort(abs_sv)[::-1][:10]
        result  = [{"feature": f"F{int(i)}", "value": round(float(sv[i]), 4), "abs": round(float(abs_sv[i]), 4)} for i in top_idx]
        print(f"SHAP done. Top: {result[0]}")
        return result

    except Exception as e:
        print(f"SHAP error: {e}")
        import traceback; traceback.print_exc()
        return []


def run_prediction(image_bytes: bytes) -> dict:
    load_all_models()

    img   = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224), Image.LANCZOS)
    arr   = preprocess_input(np.array(img, dtype=np.float32))
    feats = _models["extractor"].predict(np.expand_dims(arr, 0), verbose=0).reshape(1, -1)
    print(f"Features: {feats.shape}")

    if _models["pipeline"] == "kbest":
        feats = _models["selector"].transform(feats)
    elif _models["pipeline"] == "pca":
        if _models["scaler"]: feats = _models["scaler"].transform(feats)
        if _models["pca"]:    feats = _models["pca"].transform(feats)

    clf      = _models["clf"]
    pred_raw = clf.predict(feats)[0]

    if isinstance(pred_raw, (int, np.integer, np.floating, float)):
        idx        = int(round(float(pred_raw)))
        pred_class = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else "NORMAL"
    else:
        pred_class = str(pred_raw).strip()
        if pred_class not in CLASS_NAMES: pred_class = "NORMAL"

    if hasattr(clf, "predict_proba"):
        proba_raw = clf.predict_proba(feats)[0]
        if hasattr(clf, "classes_"):
            prob_map = {}
            for c, p in zip(clf.classes_, proba_raw):
                if isinstance(c, (int, np.integer, np.floating, float)):
                    key = CLASS_NAMES[int(round(float(c)))] if 0 <= int(round(float(c))) < len(CLASS_NAMES) else str(c)
                else:
                    key = str(c).strip()
                prob_map[key] = float(p)
        else:
            prob_map = {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba_raw) if i < len(CLASS_NAMES)}
        sorted_cls = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)
        top1_class, top1_conf = sorted_cls[0]
        top2_class = sorted_cls[1][0] if len(sorted_cls) > 1 else None
        top2_conf  = sorted_cls[1][1] if len(sorted_cls) > 1 else 0.0
    else:
        top1_class, top1_conf = pred_class, 1.0
        top2_class, top2_conf = None, 0.0
        prob_map = {pred_class: 1.0}

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
        "shap_values":          compute_shap(feats, top1_class),
    }


@app.get("/")
def root():
    return {"status": "ok", "service": "OcularAI"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": _models.get("ready", False),
            "pipeline": _models.get("pipeline", "not loaded"), "classes": CLASS_NAMES,
            "claude_validate": bool(ANTHROPIC_API_KEY)}

@app.get("/debug")
def debug():
    files = {n: {"downloaded": os.path.exists(os.path.join(MODEL_DIR, n)),
                 "size_mb": round(os.path.getsize(os.path.join(MODEL_DIR,n))/1024/1024,2)
                 if os.path.exists(os.path.join(MODEL_DIR,n)) else 0}
             for n in GDRIVE_FILES}
    return {"files": files, "models_loaded": _models.get("ready", False),
            "pipeline": _models.get("pipeline"), "class_order": dict(enumerate(CLASS_NAMES))}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files accepted.")
    contents = await file.read()
    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(400, "Image too large. Max 15 MB.")

    is_oct, reason = await is_retinal_oct(contents, file.content_type or "image/jpeg")
    if not is_oct:
        return JSONResponse({"success": False, "is_retinal": False,
            "message": "Not a retinal OCT scan. Please upload a valid OCT image."})
    try:
        result = run_prediction(contents)
        return JSONResponse({"success": True, "is_retinal": True, **result})
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        import traceback; print(traceback.format_exc())
        raise HTTPException(500, f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))