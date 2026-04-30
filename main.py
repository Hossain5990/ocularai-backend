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
# তোমার Google Drive File IDs এখানে বসাও
# Drive → file এ right click → Share → Anyone with link
# Link: drive.google.com/file/d/THIS_PART/view
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

CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL", "MH", "CSR", "AMD", "BRVO"]

DISEASE_INFO = {
    "CNV":    {"full": "Choroidal Neovascularization", "severity": "High",
               "desc": "Abnormal blood vessel growth beneath the retina causing rapid central vision loss. Requires urgent anti-VEGF therapy."},
    "DME":    {"full": "Diabetic Macular Edema", "severity": "High",
               "desc": "Fluid in the macula due to diabetic vascular leakage. Leading cause of vision loss in diabetic patients. Anti-VEGF or laser treatment required."},
    "DRUSEN": {"full": "Drusen Deposits", "severity": "Moderate",
               "desc": "Yellow deposits under the retina — an early sign of AMD. Monitor closely; AREDS2 supplements may slow progression."},
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

# Global model cache
_models = {}


def download_file(filename: str) -> str:
    """Download from Google Drive if not cached."""
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        print(f"✅ Using cached: {filename}")
        return path

    file_id = GDRIVE_FILES.get(filename)
    if not file_id or "YOUR_" in file_id:
        raise RuntimeError(
            f"❌ File ID not set for '{filename}'. "
            "Open main.py and fill in GDRIVE_FILES."
        )

    print(f"⬇ Downloading {filename} ...")
    url = f"https://drive.google.com/uc?id={file_id}"
    out = gdown.download(url, path, quiet=False, fuzzy=True)
    if not out or not os.path.exists(path):
        raise RuntimeError(
            f"❌ Failed to download '{filename}'. "
            "Check that the file is shared as 'Anyone with the link'."
        )
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"✅ Downloaded {filename} ({size_mb:.1f} MB)")
    return path


def load_pkl(filename: str):
    """Load a pickle file, return None if file ID not set."""
    try:
        path = download_file(filename)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"✅ Loaded {filename}: {type(obj).__name__}")
        return obj
    except RuntimeError as e:
        print(f"⚠ Skipping {filename}: {e}")
        return None


def build_feature_extractor(full_model):
    """Extract features from GlobalAveragePooling layer."""
    # Try to find GAP layer by name
    gap_layer = None
    for layer in reversed(full_model.layers):
        name = layer.name.lower()
        if "global_average" in name or "avg_pool" in name or "gap" in name:
            gap_layer = layer
            break

    if gap_layer is None:
        # Fallback: use second-to-last layer
        gap_layer = full_model.layers[-2]
        print(f"⚠ No GAP layer found, using: {gap_layer.name}")
    else:
        print(f"✅ Using GAP layer: {gap_layer.name}, shape: {gap_layer.output_shape}")

    return Model(inputs=full_model.input, outputs=gap_layer.output)


def load_all_models():
    """Load all models once and cache in _models dict."""
    if _models.get("ready"):
        return

    print("=" * 50)
    print("🔄 Loading all models...")
    print("=" * 50)

    # 1. Load DenseNet121
    h5_path = download_file("densenet121_finetuned.h5")
    full_model = load_model(h5_path)
    print(f"✅ Full model loaded. Layers: {len(full_model.layers)}")
    _models["extractor"] = build_feature_extractor(full_model)

    # 2. Load classifier (required)
    _models["clf"] = load_pkl("best_clf.pkl")
    if _models["clf"] is None:
        raise RuntimeError("best_clf.pkl is required but could not be loaded.")

    # 3. Load optional pipeline components
    _models["scaler"]   = load_pkl("scaler.pkl")
    _models["selector"] = load_pkl("selector.pkl")
    _models["pca"]      = load_pkl("pca.pkl")

    # 4. Detect pipeline type
    sel = _models["selector"]
    pca = _models["pca"]

    if sel is not None and hasattr(sel, "transform"):
        _models["reducer"] = sel
        _models["reducer_name"] = "SelectKBest"
    elif pca is not None and hasattr(pca, "transform"):
        _models["reducer"] = pca
        _models["reducer_name"] = "PCA"
    else:
        _models["reducer"] = None
        _models["reducer_name"] = "None"

    print(f"✅ Pipeline: scaler={_models['scaler'] is not None} | "
          f"reducer={_models['reducer_name']} | "
          f"clf={type(_models['clf']).__name__}")
    print("🚀 All models ready!")
    _models["ready"] = True


def predict_image(image_bytes: bytes) -> dict:
    """Full pipeline: bytes → disease prediction."""
    load_all_models()

    # Step 1: Preprocess image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    x = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

    # Step 2: Feature extraction
    feats = _models["extractor"].predict(x, verbose=0)
    feats = feats.reshape(1, -1)
    print(f"Features shape: {feats.shape}")

    # Step 3: Scale (if scaler available)
    if _models["scaler"] is not None:
        feats = _models["scaler"].transform(feats)
        print("✅ Scaled")

    # Step 4: Reduce (if reducer available)
    if _models["reducer"] is not None:
        feats = _models["reducer"].transform(feats)
        print(f"✅ Reduced via {_models['reducer_name']}, shape: {feats.shape}")

    # Step 5: Classify
    clf = _models["clf"]
    pred_raw = clf.predict(feats)[0]
    print(f"Raw prediction: {pred_raw} (type: {type(pred_raw).__name__})")

    # Handle both int index and string class name
    if isinstance(pred_raw, (int, np.integer)):
        pred_class = CLASS_NAMES[int(pred_raw)] if int(pred_raw) < len(CLASS_NAMES) else str(pred_raw)
    elif isinstance(pred_raw, (str, np.str_)):
        pred_class = str(pred_raw)
        # normalize e.g. "NORMAL" already correct
    else:
        pred_class = str(pred_raw)

    # Probabilities
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(feats)[0]
        # Build class→prob mapping
        if hasattr(clf, "classes_"):
            classes = clf.classes_
            prob_map = {}
            for c, p in zip(classes, proba):
                key = CLASS_NAMES[int(c)] if isinstance(c, (int, np.integer)) else str(c)
                prob_map[key] = float(p)
        else:
            prob_map = {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba) if i < len(CLASS_NAMES)}

        sorted_classes = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)
        top1_class = sorted_classes[0][0]
        top1_conf  = sorted_classes[0][1]
        top2_class = sorted_classes[1][0] if len(sorted_classes) > 1 else None
        top2_conf  = sorted_classes[1][1] if len(sorted_classes) > 1 else 0.0
    else:
        # No probabilities (e.g. LinearSVC)
        top1_class = pred_class
        top1_conf  = 1.0
        top2_class = None
        top2_conf  = 0.0
        prob_map   = {pred_class: 1.0}

    info = DISEASE_INFO.get(top1_class, DISEASE_INFO.get("NORMAL"))

    return {
        "disease":              top1_class,
        "confidence":           round(top1_conf, 4),
        "full_name":            info["full"],
        "severity":             info["severity"],
        "description":          info["desc"],
        "secondary":            top2_class,
        "secondary_confidence": round(top2_conf, 4),
        "all_probabilities":    {k: round(v, 4) for k, v in prob_map.items()},
    }


# ── Routes ──────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "OcularAI Eye Disease Detection API"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": _models.get("ready", False),
        "pipeline": _models.get("reducer_name", "not loaded"),
    }


@app.get("/debug")
def debug():
    """Shows which files are downloaded — useful for troubleshooting."""
    files = {}
    for name in GDRIVE_FILES:
        path = os.path.join(MODEL_DIR, name)
        files[name] = {
            "id_set": "YOUR_" not in GDRIVE_FILES[name],
            "downloaded": os.path.exists(path),
            "size_mb": round(os.path.getsize(path) / 1024 / 1024, 2) if os.path.exists(path) else 0,
        }
    return {"files": files, "models_loaded": _models.get("ready", False)}


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
        tb = traceback.format_exc()
        print(f"❌ Prediction error:\n{tb}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")