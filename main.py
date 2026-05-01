import os
import io
import pickle
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════
# Google Drive File IDs — তোমার actual IDs বসাও
# ══════════════════════════════════════════════════════
GDRIVE_FILES = {
    "densenet121_finetuned.h5": "1FoVidXa3rt5ANpSPNzUrxwCN1Xv9Wg_M",
    "best_clf.pkl":             "1jBE3tYjqG_nWw5Y7KRazD1SvDdHrhHgZ",
    "selector.pkl":             "1vFyw9karH1eGttLdfP-N0aK2uoXHOJpf",
    "scaler.pkl":               "156xcXLRkd4h2CWvBPIjZQ-MYu4RoJS3R",
    "pca.pkl":                  "1fNf7Hp9Yk7yuooqohkiZ5sAmTB_8ut2e",
}

MODEL_DIR = "/tmp/ocularai_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════
# Notebook থেকে পাওয়া exact pipeline:
#
# feat_extractor = model.layers[-3].output  → 256-dim features
#
# Best model = SVM + SelectKBest (95.86%)
# Pipeline:  image → DenseNet[-3] → 256 feats → SelectKBest → SVM
#
# PCA pipeline (backup):
# image → DenseNet[-3] → 256 feats → StandardScaler → PCA(200) → SVM
# ══════════════════════════════════════════════════════

CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL", "MH", "CSR", "AMD", "BRVO"]

DISEASE_INFO = {
    "CNV":    {"full": "Choroidal Neovascularization", "severity": "High",
               "desc": "Abnormal blood vessel growth beneath the retina causing rapid central vision loss. Requires urgent anti-VEGF therapy."},
    "DME":    {"full": "Diabetic Macular Edema", "severity": "High",
               "desc": "Fluid in the macula due to diabetic vascular leakage. Leading cause of vision loss in diabetic patients. Anti-VEGF or laser treatment required."},
    "DRUSEN": {"full": "Drusen Deposits", "severity": "Moderate",
               "desc": "Yellow deposits under the retina — early sign of AMD. Monitor closely; AREDS2 supplements may slow progression."},
    "NORMAL": {"full": "Normal Healthy Retina", "severity": "None",
               "desc": "No pathological changes detected. Retinal structure appears intact. Continue routine annual eye check-ups."},
    "MH":     {"full": "Macular Hole", "severity": "High",
               "desc": "Full-thickness defect in the central retina. Vitrectomy surgery is highly effective when performed early."},
    "CSR":    {"full": "Central Serous Retinopathy", "severity": "Moderate",
               "desc": "Subretinal fluid accumulation often related to stress. Most cases resolve within 3 months. Chronic cases need photodynamic therapy."},
    "AMD":    {"full": "Age-related Macular Degeneration", "severity": "High",
               "desc": "Progressive macular degeneration. Wet AMD needs urgent anti-VEGF injections; dry AMD managed with supplements."},
    "BRVO":   {"full": "Branch Retinal Vein Occlusion", "severity": "High",
               "desc": "Retinal vein blockage causing hemorrhage and macular edema. Treated with anti-VEGF and laser photocoagulation."},
}

_models = {}


def download_file(filename: str) -> str:
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        print(f"✅ Cached: {filename}")
        return path
    file_id = GDRIVE_FILES.get(filename)
    if not file_id or "YOUR_" in file_id:
        raise RuntimeError(f"File ID not set for '{filename}'.")
    print(f"⬇ Downloading {filename} ...")
    url = f"https://drive.google.com/uc?id={file_id}"
    out = gdown.download(url, path, quiet=False)
    if not out or not os.path.exists(path):
        raise RuntimeError(f"Failed to download '{filename}'. Check sharing settings.")
    print(f"✅ Downloaded {filename} ({os.path.getsize(path)/1024/1024:.1f} MB)")
    return path


def load_pkl(filename: str):
    try:
        path = download_file(filename)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"✅ Loaded {filename}: {type(obj).__name__}")
        return obj
    except Exception as e:
        print(f"⚠ Could not load {filename}: {e}")
        return None


def load_all_models():
    if _models.get("ready"):
        return

    print("=" * 55)
    print("🔄 Loading all models...")
    print("=" * 55)

    # ── 1. Load full DenseNet121 model ──────────────────────
    h5_path = download_file("densenet121_finetuned.h5")
    full_model = load_model(h5_path)
    total_layers = len(full_model.layers)
    print(f"✅ Full model loaded. Total layers: {total_layers}")

    # ── 2. Build feature extractor ──────────────────────────
    # Notebook: feat_extractor = Model(inputs, outputs=model.layers[-3].output)
    # layers[-3] = Dense(256) output → 256-dim feature vector
    target_layer = full_model.layers[-3]
    print(f"✅ Feature layer: {target_layer.name} (layers[-3])")

    try:
        shape = tuple(target_layer.output.shape)
        print(f"✅ Feature shape: {shape}")
    except Exception:
        pass

    _models["extractor"] = Model(
        inputs=full_model.input,
        outputs=target_layer.output
    )

    # ── 3. Load ML components ───────────────────────────────
    _models["clf"]      = load_pkl("best_clf.pkl")   # SVM (required)
    _models["selector"] = load_pkl("selector.pkl")   # SelectKBest (best pipeline)
    _models["scaler"]   = load_pkl("scaler.pkl")     # StandardScaler (PCA pipeline)
    _models["pca"]      = load_pkl("pca.pkl")        # PCA (backup pipeline)

    if _models["clf"] is None:
        raise RuntimeError("best_clf.pkl failed to load. Cannot predict.")

    # ── 4. Determine pipeline ───────────────────────────────
    # Best = SelectKBest pipeline (no scaler needed)
    # Backup = Scaler → PCA pipeline
    if _models["selector"] is not None:
        _models["pipeline"] = "kbest"
        print("✅ Pipeline: DenseNet[-3](256) → SelectKBest → SVM")
    elif _models["scaler"] is not None and _models["pca"] is not None:
        _models["pipeline"] = "pca"
        print("✅ Pipeline: DenseNet[-3](256) → StandardScaler → PCA → SVM")
    else:
        _models["pipeline"] = "direct"
        print("⚠ Pipeline: DenseNet[-3](256) → SVM (direct, no reduction)")

    print("🚀 All models ready!")
    _models["ready"] = True


def predict_image(image_bytes: bytes) -> dict:
    load_all_models()

    # ── Step 1: Preprocess image ────────────────────────────
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    x = np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)

    # ── Step 2: DenseNet[-3] → 256-dim features ─────────────
    feats = _models["extractor"].predict(x, verbose=0)
    feats = feats.reshape(1, -1)
    print(f"Features shape: {feats.shape}")  # should be (1, 256)

    # ── Step 3: Apply pipeline ──────────────────────────────
    pipeline = _models["pipeline"]

    if pipeline == "kbest":
        # Best pipeline: SelectKBest only (no scaler)
        feats = _models["selector"].transform(feats)
        print(f"✅ SelectKBest → shape: {feats.shape}")

    elif pipeline == "pca":
        # Backup: StandardScaler → PCA
        feats = _models["scaler"].transform(feats)
        feats = _models["pca"].transform(feats)
        print(f"✅ Scaler+PCA → shape: {feats.shape}")

    # else: direct (no reduction)

    # ── Step 4: SVM predict ─────────────────────────────────
    clf = _models["clf"]
    pred_raw = clf.predict(feats)[0]
    print(f"Raw prediction: {pred_raw}")

    # Handle int index or string class name
    if isinstance(pred_raw, (int, np.integer)):
        pred_class = CLASS_NAMES[int(pred_raw)] if int(pred_raw) < len(CLASS_NAMES) else "NORMAL"
    else:
        pred_class = str(pred_raw)

    # ── Step 5: Probabilities ───────────────────────────────
    if hasattr(clf, "predict_proba"):
        proba_raw = clf.predict_proba(feats)[0]
        if hasattr(clf, "classes_"):
            prob_map = {}
            for c, p in zip(clf.classes_, proba_raw):
                key = CLASS_NAMES[int(c)] if isinstance(c, (int, np.integer)) else str(c)
                prob_map[key] = float(p)
        else:
            prob_map = {CLASS_NAMES[i]: float(p)
                        for i, p in enumerate(proba_raw) if i < len(CLASS_NAMES)}

        sorted_cls = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)
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
        "pipeline_used":        _models.get("pipeline", "unknown"),
    }


# ── Routes ──────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "OcularAI Eye Disease Detection API"}


@app.get("/health")
def health():
    return {
        "status":        "healthy",
        "model_loaded":  _models.get("ready", False),
        "pipeline":      _models.get("pipeline", "not loaded"),
    }


@app.get("/debug")
def debug():
    files = {}
    for name in GDRIVE_FILES:
        path = os.path.join(MODEL_DIR, name)
        files[name] = {
            "id_set":      "YOUR_" not in GDRIVE_FILES[name],
            "downloaded":  os.path.exists(path),
            "size_mb":     round(os.path.getsize(path)/1024/1024, 2) if os.path.exists(path) else 0,
        }
    return {
        "files":         files,
        "models_loaded": _models.get("ready", False),
        "pipeline":      _models.get("pipeline", "not loaded"),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files accepted.")
    contents = await file.read()
    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large. Max 15 MB.")
    try:
        result = predict_image(contents)
        return JSONResponse(content={"success": True, **result})
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
