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
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

app = FastAPI(title="OcularAI — Eye Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Vercel frontend URL later restrict korte paro
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Google Drive File IDs — তোমার Drive থেকে নাও
# (নিচে কীভাবে পাবে সেটা README তে আছে)
# ─────────────────────────────────────────────
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
               "desc": "Subretinal fluid accumulation often related to stress. Most acute cases resolve within 3 months. Chronic cases may need photodynamic therapy."},
    "AMD":    {"full": "Age-related Macular Degeneration", "severity": "High",
               "desc": "Progressive macular degeneration. Wet AMD needs urgent anti-VEGF injections; dry AMD managed with supplements."},
    "BRVO":   {"full": "Branch Retinal Vein Occlusion", "severity": "High",
               "desc": "Retinal vein blockage causing hemorrhage and macular edema. Treated with anti-VEGF injections and laser photocoagulation."},
}

# ─────────────────────────────────────────────
# Global model objects (lazy loaded)
# ─────────────────────────────────────────────
feature_extractor = None
clf = None
selector = None
scaler = None
pca = None
use_kbest = None   # True = SelectKBest, False = PCA


def download_if_missing(filename: str) -> str:
    """Download file from Google Drive if not already cached in /tmp."""
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        file_id = GDRIVE_FILES.get(filename)
        if not file_id or file_id.startswith("YOUR_"):
            raise RuntimeError(
                f"Google Drive file ID for '{filename}' is not set. "
                "Please update GDRIVE_FILES in main.py."
            )
        print(f"⬇  Downloading {filename} from Google Drive …")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)
        print(f"✅  {filename} saved to {path}")
    return path


def load_models():
    """Load all model components (called once on first request)."""
    global feature_extractor, clf, selector, scaler, pca, use_kbest

    if feature_extractor is not None:
        return  # already loaded

    print("🔄  Loading models …")

    # 1. DenseNet121 — feature extractor (up to GlobalAveragePooling layer)
    h5_path = download_if_missing("densenet121_finetuned.h5")
    full_model = load_model(h5_path)

    # Build feature extractor from the trained model
    # We extract from the GlobalAveragePooling2D layer output
    gap_layer = None
    for layer in reversed(full_model.layers):
        if "global_average_pooling" in layer.name or "avg_pool" in layer.name:
            gap_layer = layer
            break
    if gap_layer is None:
        # fallback: use second-to-last layer
        gap_layer = full_model.layers[-2]

    feature_extractor = Model(
        inputs=full_model.input,
        outputs=gap_layer.output
    )
    print(f"✅  Feature extractor ready — output shape: {gap_layer.output_shape}")

    # 2. Classical ML components
    with open(download_if_missing("best_clf.pkl"), "rb") as f:
        clf = pickle.load(f)

    with open(download_if_missing("selector.pkl"), "rb") as f:
        selector = pickle.load(f)

    with open(download_if_missing("scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    with open(download_if_missing("pca.pkl"), "rb") as f:
        pca = pickle.load(f)

    # Detect pipeline type (same logic as your notebook)
    use_kbest = hasattr(selector, 'scores_')   # SelectKBest has scores_
    print(f"✅  Pipeline: {'SelectKBest' if use_kbest else 'PCA'} + {type(clf).__name__}")
    print("🚀  All models loaded — ready to predict!")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Resize and preprocess image for DenseNet121."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)          # DenseNet121 preprocessing
    return np.expand_dims(arr, axis=0)   # shape (1, 224, 224, 3)


def extract_and_predict(image_bytes: bytes) -> dict:
    """Full pipeline: image → features → scale → select/pca → SVM → result."""
    load_models()

    # 1. Preprocess
    x = preprocess_image(image_bytes)

    # 2. DenseNet121 feature extraction
    feats = feature_extractor.predict(x, verbose=0)   # shape (1, 1024)
    feats_flat = feats.reshape(1, -1)

    # 3. Scale
    feats_scaled = scaler.transform(feats_flat)

    # 4. Dimensionality reduction (match your notebook's pipeline)
    if use_kbest:
        feats_reduced = selector.transform(feats_scaled)
    else:
        feats_reduced = pca.transform(feats_scaled)

    # 5. SVM classification
    pred_idx = clf.predict(feats_reduced)[0]
    proba = clf.predict_proba(feats_reduced)[0]  # shape (n_classes,)

    # Map index → class name
    # Your notebook uses folder-based class ordering; CLASS_NAMES matches that order
    pred_class = CLASS_NAMES[pred_idx] if isinstance(pred_idx, (int, np.integer)) else str(pred_idx)

    # Sort probabilities for top-2
    sorted_idx = np.argsort(proba)[::-1]
    top1_class = CLASS_NAMES[sorted_idx[0]] if isinstance(sorted_idx[0], (int, np.integer)) else pred_class
    top1_conf  = float(proba[sorted_idx[0]])
    top2_class = CLASS_NAMES[sorted_idx[1]] if len(sorted_idx) > 1 else None
    top2_conf  = float(proba[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0

    info = DISEASE_INFO.get(top1_class, {})

    return {
        "disease":               top1_class,
        "confidence":            round(top1_conf, 4),
        "full_name":             info.get("full", top1_class),
        "severity":              info.get("severity", "Unknown"),
        "description":           info.get("desc", ""),
        "secondary":             top2_class,
        "secondary_confidence":  round(top2_conf, 4),
        "all_probabilities":     {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(proba)},
    }


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "OcularAI Eye Disease Detection API"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": feature_extractor is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Basic file validation
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted.")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:   # 10 MB limit
        raise HTTPException(status_code=400, detail="Image too large. Max 10 MB.")

    try:
        result = extract_and_predict(contents)
        return JSONResponse(content={"success": True, **result})
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
