# import os, io, gc, pickle, base64, httpx
# import gdown
# import numpy as np
# from PIL import Image
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse

# # ══════════════════════════════════════════════════════
# # TensorFlow — Memory-safe import
# # Render free tier = 512MB RAM. TF eager + full GPU ops
# # wastes memory. Limit threads + disable GPU ops.
# # ══════════════════════════════════════════════════════
# os.environ["TF_CPP_MIN_LOG_LEVEL"]        = "3"       # suppress TF logs
# os.environ["OMP_NUM_THREADS"]             = "1"       # 1 CPU thread
# os.environ["TF_NUM_INTRAOP_THREADS"]      = "1"
# os.environ["TF_NUM_INTEROP_THREADS"]      = "1"
# os.environ["TF_ENABLE_ONEDNN_OPTS"]       = "0"

# import tensorflow as tf

# # Limit TF to use minimum memory — don't pre-allocate a big pool
# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)

# from tensorflow.keras.applications.densenet import preprocess_input
# from tensorflow.keras.models import load_model, Model


# # ══════════════════════════════════════════════════════
# # App setup
# # ══════════════════════════════════════════════════════
# app = FastAPI(title="OcularAI Eye Disease Detection API")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ══════════════════════════════════════════════════════
# # Config
# # ══════════════════════════════════════════════════════
# GDRIVE_FILES = {
#     "densenet121_finetuned.h5": "1FoVidXa3rt5ANpSPNzUrxwCN1Xv9Wg_M",
#     "best_clf.pkl":             "1jBE3tYjqG_nWw5Y7KRazD1SvDdHrhHgZ",
#     "selector.pkl":             "1vFyw9karH1eGttLdfP-N0aK2uoXHOJpf",
#     "scaler.pkl":               "156xcXLRkd4h2CWvBPIjZQ-MYu4RoJS3R",
#     "pca.pkl":                  "1fNf7Hp9Yk7yuooqohkiZ5sAmTB_8ut2e",
# }

# ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# MODEL_DIR = "/tmp/ocularai_models"
# os.makedirs(MODEL_DIR, exist_ok=True)

# # Notebook STEP 3 → CLASS_NAMES = sorted(train_cnt.keys())
# # Dataset: AMD, CNV, CSR, DME, DR, DRUSEN, MH, NORMAL → alphabetical
# CLASS_NAMES = ["AMD", "CNV", "CSR", "DME", "DR", "DRUSEN", "MH", "NORMAL"]

# DISEASE_INFO = {
#     "AMD":    {
#         "full":     "Age-related Macular Degeneration",
#         "severity": "High",
#         "desc":     (
#             "Progressive degeneration of the macula. "
#             "Wet AMD needs urgent anti-VEGF injections; "
#             "dry AMD managed with AREDS2 supplements."
#         ),
#     },
#     "CNV":    {
#         "full":     "Choroidal Neovascularization",
#         "severity": "High",
#         "desc":     (
#             "Abnormal blood vessel growth beneath the retina "
#             "causing rapid central vision loss. "
#             "Requires urgent anti-VEGF therapy."
#         ),
#     },
#     "CSR":    {
#         "full":     "Central Serous Retinopathy",
#         "severity": "Moderate",
#         "desc":     (
#             "Subretinal fluid accumulation often related to stress. "
#             "Most acute cases resolve in 3 months. "
#             "Chronic cases may need photodynamic therapy."
#         ),
#     },
#     "DME":    {
#         "full":     "Diabetic Macular Edema",
#         "severity": "High",
#         "desc":     (
#             "Fluid in the macula due to diabetic vascular leakage. "
#             "Leading cause of vision loss in diabetics. "
#             "Anti-VEGF or laser treatment required."
#         ),
#     },
#     "DR":     {
#         "full":     "Diabetic Retinopathy",
#         "severity": "High",
#         "desc":     (
#             "Damage to retinal blood vessels caused by long-term diabetes. "
#             "Can progress to blindness. "
#             "Laser therapy, anti-VEGF, and strict blood sugar control needed."
#         ),
#     },
#     "DRUSEN": {
#         "full":     "Drusen Deposits",
#         "severity": "Moderate",
#         "desc":     (
#             "Yellow lipid deposits beneath the retina — early sign of AMD. "
#             "Regular monitoring essential. "
#             "AREDS2 supplements may slow progression."
#         ),
#     },
#     "MH":     {
#         "full":     "Macular Hole",
#         "severity": "High",
#         "desc":     (
#             "Full-thickness defect in the central retina. "
#             "Pars plana vitrectomy is highly effective (>90% success) "
#             "when performed early."
#         ),
#     },
#     "NORMAL": {
#         "full":     "Normal Healthy Retina",
#         "severity": "None",
#         "desc":     (
#             "No pathological changes detected. "
#             "Retinal layers appear intact. "
#             "Continue routine annual eye examinations."
#         ),
#     },
# }

# # ══════════════════════════════════════════════════════
# # Global model store — loaded once, reused forever
# # ══════════════════════════════════════════════════════
# _models: dict = {}


# # ══════════════════════════════════════════════════════
# # Step 1 — Claude Vision: is this a retinal OCT scan?
# # ══════════════════════════════════════════════════════
# async def is_retinal_oct(image_bytes: bytes, media_type: str) -> tuple[bool, str]:
#     if not ANTHROPIC_API_KEY:
#         print("⚠ No ANTHROPIC_API_KEY — skipping retinal validation")
#         return True, "validation_skipped"

#     b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
#     payload = {
#         "model": "claude-sonnet-4-20250514",
#         "max_tokens": 10,           # YES/NO only — keep it tiny
#         "system": (
#             "You are a medical image classifier. "
#             "Your ONLY job: decide if the image is a retinal OCT "
#             "(Optical Coherence Tomography) scan. "
#             "Retinal OCT scans show cross-sectional layers of the retina "
#             "— typically grayscale with horizontal banded structures. "
#             "Reply with ONLY one word: YES or NO."
#         ),
#         "messages": [{
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                     "source": {
#                         "type":       "base64",
#                         "media_type": media_type,
#                         "data":       b64,
#                     },
#                 },
#                 {
#                     "type": "text",
#                     "text": "Is this a retinal OCT scan? Reply YES or NO only.",
#                 },
#             ],
#         }],
#     }

#     try:
#         async with httpx.AsyncClient(timeout=20) as client:
#             resp = await client.post(
#                 "https://api.anthropic.com/v1/messages",
#                 headers={
#                     "x-api-key":         ANTHROPIC_API_KEY,
#                     "anthropic-version": "2023-06-01",
#                     "content-type":      "application/json",
#                 },
#                 json=payload,
#             )
#         data   = resp.json()
#         answer = data["content"][0]["text"].strip().upper()
#         print(f"Claude OCT check → {answer}")
#         return answer.startswith("YES"), answer
#     except Exception as e:
#         print(f"⚠ Claude validation error: {e} — allowing image through")
#         return True, f"validation_error: {e}"


# # ══════════════════════════════════════════════════════
# # File download helper
# # ══════════════════════════════════════════════════════
# def download_file(filename: str) -> str:
#     path = os.path.join(MODEL_DIR, filename)
#     if os.path.exists(path):
#         size_mb = os.path.getsize(path) / 1024 / 1024
#         print(f"✅ Cached: {filename} ({size_mb:.1f} MB)")
#         return path

#     file_id = GDRIVE_FILES.get(filename)
#     if not file_id or "YOUR_" in file_id:
#         raise RuntimeError(f"❌ Google Drive file ID not set for '{filename}'.")

#     print(f"⬇ Downloading {filename} from Google Drive ...")
#     out = gdown.download(
#         f"https://drive.google.com/uc?id={file_id}", path, quiet=False
#     )
#     if not out or not os.path.exists(path):
#         raise RuntimeError(
#             f"❌ Download failed for '{filename}'. "
#             "Check that the file is shared publicly on Google Drive."
#         )
#     size_mb = os.path.getsize(path) / 1024 / 1024
#     print(f"✅ Downloaded {filename} ({size_mb:.1f} MB)")
#     return path


# def load_pkl(filename: str):
#     try:
#         with open(download_file(filename), "rb") as f:
#             obj = pickle.load(f)
#         print(f"✅ Loaded {filename}: {type(obj).__name__}")
#         return obj
#     except Exception as e:
#         print(f"⚠ Skipping {filename}: {e}")
#         return None


# # ══════════════════════════════════════════════════════════════════════════
# # Model loading — memory-aware strategy for Render 512 MB
# #
# # Architecture from notebook (STEP 5 + STEP 6):
# #   DenseNet121 base
# #     → GlobalAveragePooling2D
# #     → BatchNormalization
# #     → Dense(512, relu)          layers[-5]
# #     → Dropout(0.4)              layers[-4]
# #     → Dense(256, relu)          layers[-3]  ← feature extractor output
# #     → Dropout(0.3)              layers[-2]
# #     → Dense(8, softmax)         layers[-1]
# #
# # Feature shape = (1, 256)
# # SelectKBest: K = min(700, 256) = 256 → effectively a no-op selector
# #              but we still call .transform() to be safe
# #
# # Memory breakdown on Render:
# #   densenet121_finetuned.h5  ≈ 30–50 MB on disk, ~200 MB in TF RAM
# #   sklearn SVM pkl           ≈ 10–50 MB (support vectors)
# #   selector / scaler / pca   ≈ <5 MB each
# #   TF runtime overhead       ≈ 100–150 MB
# #   Total est.                ≈ 350–450 MB  (fits in 512 MB with care)
# #
# # Optimizations applied:
# #   1. Load model with compile=False → skips optimizer state (~10 MB)
# #   2. Build extractor (sub-model) and del full model → free softmax head RAM
# #   3. gc.collect() after every heavy load
# #   4. predict with batch_size=1, verbose=0
# # ══════════════════════════════════════════════════════════════════════════
# def load_all_models() -> None:
#     if _models.get("ready"):
#         return

#     print("=" * 55)
#     print("🔄 Loading models (memory-optimised for Render 512 MB)...")

#     # ── 1. Load full DenseNet model ──────────────────────────────────────
#     h5_path    = download_file("densenet121_finetuned.h5")
#     full_model = load_model(h5_path, compile=False)   # compile=False saves ~10 MB
#     print(f"✅ Full model loaded | layers: {len(full_model.layers)}")

#     # ── 2. Verify layer[-3] is the Dense(256) we expect ──────────────────
#     # NOTE: TF 2.16+ এ layer.output_shape কাজ করে না (AttributeError)।
#     #       layer.output.shape ব্যবহার করতে হবে — এটা সবসময় কাজ করে।
#     feat_layer   = full_model.layers[-3]
#     feat_shape   = feat_layer.output.shape   # e.g. (None, 256)
#     feat_dim     = int(feat_shape[-1])
#     print(
#         f"✅ Feature layer  : index=-3  name={feat_layer.name}  "
#         f"output_shape={feat_shape}"
#     )
#     if feat_dim != 256:
#         print(
#             f"⚠ WARNING: Expected 256-dim features at layers[-3], "
#             f"got {feat_dim}. Check model architecture!"
#         )

#     # ── 3. Build lightweight extractor, then delete full model ───────────
#     _models["extractor"] = Model(
#         inputs=full_model.input, outputs=feat_layer.output
#     )
#     del full_model          # release softmax head memory
#     gc.collect()
#     print("✅ Extractor built | full model freed from RAM")

#     # ── 4. Load sklearn artifacts ─────────────────────────────────────────
#     clf      = load_pkl("best_clf.pkl")
#     selector = load_pkl("selector.pkl")
#     scaler   = load_pkl("scaler.pkl")
#     pca      = load_pkl("pca.pkl")
#     gc.collect()

#     if clf is None:
#         raise RuntimeError("best_clf.pkl failed to load. Cannot serve predictions.")

#     _models["clf"]      = clf
#     _models["selector"] = selector
#     _models["scaler"]   = scaler
#     _models["pca"]      = pca

#     # ── 5. Determine pipeline from what loaded successfully ───────────────
#     #
#     #  Notebook trains two pipelines:
#     #    kbest : features → SelectKBest(k=256) → SVM
#     #    pca   : features → StandardScaler → PCA(200) → SVM
#     #
#     #  best_clf.pkl = whichever SVM won. We check selector first because
#     #  if selector loaded, that pipeline was used with best_clf.
#     #
#     if selector is not None:
#         _models["pipeline"] = "kbest"
#         print(
#             f"✅ Pipeline: DenseNet→Dense(256)→SelectKBest(k={feat_dim})→SVM"
#         )
#     elif scaler is not None and pca is not None:
#         _models["pipeline"] = "pca"
#         print("✅ Pipeline: DenseNet→Dense(256)→Scaler→PCA(200)→SVM")
#     else:
#         _models["pipeline"] = "direct"
#         print("✅ Pipeline: DenseNet→Dense(256)→SVM (no dim-reduction)")

#     # ── 6. Warm-up inference (catches shape mismatches at startup) ────────
#     dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
#     dummy_feat = _models["extractor"].predict(dummy, batch_size=1, verbose=0)
#     dummy_feat = _apply_dim_reduction(dummy_feat)
#     _models["clf"].predict(dummy_feat)
#     print(f"✅ Warm-up OK | feature shape after reduction: {dummy_feat.shape}")

#     print(f"🚀 Ready!  Classes: {CLASS_NAMES}")
#     _models["ready"] = True


# # ══════════════════════════════════════════════════════
# # Dimensionality reduction (shared by warmup + predict)
# # ══════════════════════════════════════════════════════
# def _apply_dim_reduction(feats: np.ndarray) -> np.ndarray:
#     """Apply SelectKBest / Scaler+PCA / nothing, based on loaded pipeline."""
#     pipeline = _models.get("pipeline", "direct")

#     if pipeline == "kbest" and _models["selector"] is not None:
#         feats = _models["selector"].transform(feats)

#     elif pipeline == "pca":
#         if _models["scaler"] is not None:
#             feats = _models["scaler"].transform(feats)
#         if _models["pca"] is not None:
#             feats = _models["pca"].transform(feats)

#     return feats


# # ══════════════════════════════════════════════════════
# # Prediction
# # ══════════════════════════════════════════════════════
# def run_prediction(image_bytes: bytes) -> dict:
#     load_all_models()   # no-op if already loaded

#     # ── Preprocess image ─────────────────────────────────────────────────
#     img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     img = img.resize((224, 224), Image.LANCZOS)
#     arr = preprocess_input(np.array(img, dtype=np.float32))   # DenseNet norm
#     x   = np.expand_dims(arr, axis=0)                         # (1,224,224,3)

#     # ── Extract 256-dim features ─────────────────────────────────────────
#     feats = _models["extractor"].predict(x, batch_size=1, verbose=0)
#     feats = feats.reshape(1, -1)                               # (1, 256)
#     print(f"Raw features shape: {feats.shape}")

#     # ── Dimensionality reduction ─────────────────────────────────────────
#     feats = _apply_dim_reduction(feats)
#     print(f"Reduced features shape: {feats.shape}")

#     # ── SVM predict ──────────────────────────────────────────────────────
#     clf      = _models["clf"]
#     pred_raw = clf.predict(feats)[0]

#     # pred_raw may be int (label-encoded) or string class name
#     if isinstance(pred_raw, (int, np.integer)):
#         idx        = int(pred_raw)
#         pred_class = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "NORMAL"
#     else:
#         pred_class = str(pred_raw)

#     # ── Probabilities ─────────────────────────────────────────────────────
#     if hasattr(clf, "predict_proba"):
#         proba_raw = clf.predict_proba(feats)[0]   # shape: (n_classes,)

#         # Map classifier class indices → CLASS_NAMES strings
#         prob_map: dict[str, float] = {}
#         if hasattr(clf, "classes_"):
#             for c, p in zip(clf.classes_, proba_raw):
#                 # c could be int index or string label
#                 if isinstance(c, (int, np.integer)):
#                     key = CLASS_NAMES[int(c)] if int(c) < len(CLASS_NAMES) else str(c)
#                 else:
#                     key = str(c)
#                 prob_map[key] = float(p)
#         else:
#             # Fallback: assume order matches CLASS_NAMES
#             for i, p in enumerate(proba_raw):
#                 if i < len(CLASS_NAMES):
#                     prob_map[CLASS_NAMES[i]] = float(p)

#         # Sort by probability descending
#         sorted_cls = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)
#         top1_class, top1_conf = sorted_cls[0]
#         top2_class = sorted_cls[1][0] if len(sorted_cls) > 1 else None
#         top2_conf  = sorted_cls[1][1] if len(sorted_cls) > 1 else 0.0

#     else:
#         # SVM without probability=True — just use argmax class
#         top1_class, top1_conf = pred_class, 1.0
#         top2_class, top2_conf = None, 0.0
#         prob_map = {pred_class: 1.0}

#     info = DISEASE_INFO.get(top1_class, DISEASE_INFO["NORMAL"])

#     return {
#         "disease":              top1_class,
#         "confidence":           round(top1_conf, 4),
#         "full_name":            info["full"],
#         "severity":             info["severity"],
#         "description":          info["desc"],
#         "secondary":            top2_class,
#         "secondary_confidence": round(top2_conf, 4),
#         "all_probabilities":    {k: round(v, 4) for k, v in prob_map.items()},
#     }


# # ══════════════════════════════════════════════════════
# # Routes
# # ══════════════════════════════════════════════════════
# @app.get("/")
# def root():
#     return {
#         "status":  "ok",
#         "service": "OcularAI — Retinal Eye Disease Detection",
#         "classes": CLASS_NAMES,
#     }


# @app.get("/health")
# def health():
#     return {
#         "status":          "healthy",
#         "model_loaded":    _models.get("ready", False),
#         "pipeline":        _models.get("pipeline", "not_loaded"),
#         "classes":         CLASS_NAMES,
#         "claude_validate": bool(ANTHROPIC_API_KEY),
#     }


# @app.get("/debug")
# def debug():
#     files = {}
#     for name in GDRIVE_FILES:
#         path = os.path.join(MODEL_DIR, name)
#         files[name] = {
#             "id_set":     "YOUR_" not in GDRIVE_FILES[name],
#             "downloaded": os.path.exists(path),
#             "size_mb":    (
#                 round(os.path.getsize(path) / 1024 / 1024, 2)
#                 if os.path.exists(path) else 0
#             ),
#         }

#     # Memory snapshot (Linux /proc only — safe to fail)
#     mem = {}
#     try:
#         with open("/proc/meminfo") as f:
#             for line in f:
#                 if line.startswith(("MemTotal", "MemAvailable", "MemFree")):
#                     key, val = line.split(":")
#                     mem[key.strip()] = val.strip()
#     except Exception:
#         pass

#     return {
#         "files":         files,
#         "models_loaded": _models.get("ready", False),
#         "pipeline":      _models.get("pipeline", "not_loaded"),
#         "class_order":   {i: c for i, c in enumerate(CLASS_NAMES)},
#         "claude_key_set": bool(ANTHROPIC_API_KEY),
#         "memory":        mem,
#     }


# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     # ── Validate file type ────────────────────────────────────────────────
#     if not file.content_type or not file.content_type.startswith("image/"):
#         raise HTTPException(400, "Only image files are accepted.")

#     contents = await file.read()

#     if len(contents) > 15 * 1024 * 1024:
#         raise HTTPException(400, "Image too large. Maximum size is 15 MB.")

#     media_type = file.content_type or "image/jpeg"

#     # ── Step 1: Claude Vision OCT validation ─────────────────────────────
#     is_oct, reason = await is_retinal_oct(contents, media_type)
#     if not is_oct:
#         return JSONResponse({
#             "success":    False,
#             "is_retinal": False,
#             "message": (
#                 "This does not appear to be a retinal OCT scan. "
#                 "Please upload a valid cross-sectional retinal OCT image. "
#                 "Regular photos, selfies, or other medical images are not supported."
#             ),
#         })

#     # ── Step 2: Disease prediction ────────────────────────────────────────
#     try:
#         result = run_prediction(contents)
#         return JSONResponse({"success": True, "is_retinal": True, **result})

#     except RuntimeError as e:
#         raise HTTPException(500, str(e))

#     except Exception as e:
#         import traceback
#         print(traceback.format_exc())
#         raise HTTPException(500, f"Prediction failed: {str(e)}")


import os, io, gc, pickle, base64, httpx, threading
import gdown
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import load_model, Model

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
MODEL_DIR         = "/tmp/ocularai_models"
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

_models: dict = {}


async def is_retinal_oct(image_bytes: bytes, media_type: str) -> tuple[bool, str]:
    if not ANTHROPIC_API_KEY:
        print("⚠ No ANTHROPIC_API_KEY — skipping retinal validation")
        return True, "validation_skipped"
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": "claude-sonnet-4-20250514", "max_tokens": 10,
        "system": ("You are a medical image classifier. "
                   "Decide if the image is a retinal OCT scan. "
                   "Reply with ONLY one word: YES or NO."),
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
            {"type": "text", "text": "Is this a retinal OCT scan? Reply YES or NO only."},
        ]}],
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY,
                         "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json=payload,
            )
        answer = resp.json()["content"][0]["text"].strip().upper()
        print(f"Claude OCT check → {answer}")
        return answer.startswith("YES"), answer
    except Exception as e:
        print(f"⚠ Claude validation error: {e} — allowing through")
        return True, f"error: {e}"


def download_file(filename: str) -> str:
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        print(f"✅ Cached: {filename} ({os.path.getsize(path)/1024/1024:.1f} MB)")
        return path
    file_id = GDRIVE_FILES.get(filename)
    if not file_id or "YOUR_" in file_id:
        raise RuntimeError(f"❌ File ID not set for '{filename}'.")
    print(f"⬇ Downloading {filename} ...")
    out = gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)
    if not out or not os.path.exists(path):
        raise RuntimeError(f"❌ Download failed for '{filename}'.")
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


def load_all_models() -> None:
    if _models.get("ready"):
        return
    print("=" * 55)
    print("🔄 Loading models...")

    full_model = load_model(download_file("densenet121_finetuned.h5"), compile=False)
    print(f"✅ Full model loaded | layers: {len(full_model.layers)}")

    feat_layer = full_model.layers[-3]
    feat_dim   = int(feat_layer.output.shape[-1])
    print(f"✅ Feature layer: {feat_layer.name} | dim: {feat_dim}")
    if feat_dim != 256:
        print(f"⚠ WARNING: Expected 256-dim, got {feat_dim}")

    _models["extractor"] = Model(inputs=full_model.input, outputs=feat_layer.output)
    del full_model
    gc.collect()
    print("✅ Extractor built | full model freed")

    clf      = load_pkl("best_clf.pkl")
    selector = load_pkl("selector.pkl")
    scaler   = load_pkl("scaler.pkl")
    pca      = load_pkl("pca.pkl")
    gc.collect()

    if clf is None:
        raise RuntimeError("best_clf.pkl failed to load.")

    _models.update(clf=clf, selector=selector, scaler=scaler, pca=pca)

    if selector is not None:
        _models["pipeline"] = "kbest"
        print(f"✅ Pipeline: DenseNet→Dense({feat_dim})→SelectKBest→SVM")
    elif scaler is not None and pca is not None:
        _models["pipeline"] = "pca"
        print(f"✅ Pipeline: DenseNet→Dense({feat_dim})→Scaler→PCA→SVM")
    else:
        _models["pipeline"] = "direct"

    dummy      = np.zeros((1, 224, 224, 3), dtype=np.float32)
    dummy_feat = _models["extractor"].predict(dummy, batch_size=1, verbose=0)
    dummy_feat = _apply_dim_reduction(dummy_feat.reshape(1, -1))
    clf.predict(dummy_feat)
    print(f"✅ Warm-up OK | reduced shape: {dummy_feat.shape}")
    print(f"🚀 Ready! Classes: {CLASS_NAMES}")
    _models["ready"] = True


def _apply_dim_reduction(feats: np.ndarray) -> np.ndarray:
    pipeline = _models.get("pipeline", "direct")
    if pipeline == "kbest" and _models.get("selector"):
        feats = _models["selector"].transform(feats)
    elif pipeline == "pca":
        if _models.get("scaler"):
            feats = _models["scaler"].transform(feats)
        if _models.get("pca"):
            feats = _models["pca"].transform(feats)
    return feats


def compute_feature_importance(clf, feats: np.ndarray, pred_class_idx: int, top_n: int = 15) -> list:
    """
    SVM-native feature importance — no SHAP library needed.
    Linear SVM  : coef_[class] × feature_values  (exact)
    RBF SVM     : (dual_coef_ @ support_vectors_) × feature_values  (approx)
    Returns same {feature, value, abs} format — frontend buildShapBar() works unchanged.
    """
    try:
        feat_vec = feats[0]
        if hasattr(clf, "coef_"):
            coef = np.array(clf.coef_)
            if coef.ndim == 1:
                coef = coef.reshape(1, -1)
            row        = coef[pred_class_idx] if coef.shape[0] > 1 else coef[0]
            importance = row * feat_vec
        elif hasattr(clf, "dual_coef_") and hasattr(clf, "support_vectors_"):
            dc         = np.array(clf.dual_coef_)
            sv         = np.array(clf.support_vectors_)
            row_idx    = min(pred_class_idx, dc.shape[0] - 1)
            w_approx   = dc[row_idx] @ sv
            importance = w_approx * feat_vec
        else:
            importance = feat_vec.copy()

        top_idx = np.argsort(np.abs(importance))[::-1][:top_n]
        return [{"feature": f"F{int(i)}",
                 "value":   round(float(importance[i]), 5),
                 "abs":     round(float(abs(importance[i])), 5)}
                for i in top_idx]
    except Exception as e:
        print(f"⚠ Feature importance error: {e}")
        return []


def run_prediction(image_bytes: bytes) -> dict:
    load_all_models()

    img  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img  = img.resize((224, 224), Image.LANCZOS)
    arr  = preprocess_input(np.array(img, dtype=np.float32))
    x    = np.expand_dims(arr, axis=0)

    feats = _models["extractor"].predict(x, batch_size=1, verbose=0).reshape(1, -1)
    print(f"Raw features: {feats.shape}")
    feats = _apply_dim_reduction(feats)
    print(f"Reduced features: {feats.shape}")

    clf      = _models["clf"]
    pred_raw = clf.predict(feats)[0]

    if isinstance(pred_raw, (int, np.integer)):
        pred_class = CLASS_NAMES[int(pred_raw)] if int(pred_raw) < len(CLASS_NAMES) else "NORMAL"
    else:
        pred_class = str(pred_raw)

    sorted_cls = [(pred_class, 1.0)]
    prob_map   = {pred_class: 1.0}
    top2_class, top2_conf = None, 0.0

    if hasattr(clf, "predict_proba"):
        proba_raw = clf.predict_proba(feats)[0]
        prob_map  = {}
        if hasattr(clf, "classes_"):
            for c, p in zip(clf.classes_, proba_raw):
                key = CLASS_NAMES[int(c)] if isinstance(c, (int, np.integer)) else str(c)
                prob_map[key] = float(p)
        else:
            prob_map = {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba_raw) if i < len(CLASS_NAMES)}
        sorted_cls = sorted(prob_map.items(), key=lambda x: x[1], reverse=True)
        pred_class = sorted_cls[0][0]
        top2_class = sorted_cls[1][0] if len(sorted_cls) > 1 else None
        top2_conf  = sorted_cls[1][1] if len(sorted_cls) > 1 else 0.0

    top1_conf = prob_map.get(pred_class, 1.0)

    pred_class_idx = 0
    if hasattr(clf, "classes_"):
        for i, c in enumerate(clf.classes_):
            c_name = CLASS_NAMES[int(c)] if isinstance(c, (int, np.integer)) else str(c)
            if c_name == pred_class:
                pred_class_idx = i
                break

    shap_values = compute_feature_importance(clf, feats, pred_class_idx, top_n=15)
    print(f"✅ XAI done | features: {len(shap_values)}")

    info = DISEASE_INFO.get(pred_class, DISEASE_INFO["NORMAL"])
    return {
        "disease":              pred_class,
        "confidence":           round(top1_conf, 4),
        "full_name":            info["full"],
        "severity":             info["severity"],
        "description":          info["desc"],
        "secondary":            top2_class,
        "secondary_confidence": round(top2_conf, 4),
        "all_probabilities":    {k: round(v, 4) for k, v in prob_map.items()},
        "shap_values":          shap_values,
    }


def _background_load():
    try:
        print("🔄 Background model loading started...")
        load_all_models()
    except Exception as e:
        import traceback
        print(f"❌ Background load FAILED: {e}")
        print(traceback.format_exc())


@app.on_event("startup")
async def startup_event():
    threading.Thread(target=_background_load, daemon=True).start()
    print("✅ App started — model loading in background thread")


@app.get("/")
def root():
    return {"status": "ok", "service": "OcularAI", "classes": CLASS_NAMES}


@app.get("/health")
def health():
    return {
        "status":          "healthy",
        "model_loaded":    _models.get("ready", False),
        "pipeline":        _models.get("pipeline", "not_loaded"),
        "classes":         CLASS_NAMES,
        "claude_validate": bool(ANTHROPIC_API_KEY),
    }


@app.get("/debug")
def debug():
    files = {}
    for name in GDRIVE_FILES:
        path = os.path.join(MODEL_DIR, name)
        files[name] = {"downloaded": os.path.exists(path),
                       "size_mb": round(os.path.getsize(path)/1024/1024, 2) if os.path.exists(path) else 0}
    mem = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(("MemTotal", "MemAvailable", "MemFree")):
                    k, v = line.split(":")
                    mem[k.strip()] = v.strip()
    except Exception:
        pass
    return {"files": files, "models_loaded": _models.get("ready", False),
            "pipeline": _models.get("pipeline", "not_loaded"),
            "class_order": {i: c for i, c in enumerate(CLASS_NAMES)}, "memory": mem}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not _models.get("ready"):
        return JSONResponse(status_code=503, content={
            "success": False,
            "message": "Models are still loading. Please wait 1-2 minutes and try again.",
        })
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are accepted.")
    contents = await file.read()
    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(400, "Image too large. Maximum 15 MB.")
    media_type = file.content_type or "image/jpeg"
    is_oct, reason = await is_retinal_oct(contents, media_type)
    if not is_oct:
        return JSONResponse({"success": False, "is_retinal": False,
            "message": "This does not appear to be a retinal OCT scan. Please upload a valid OCT image."})
    try:
        result = run_prediction(contents)
        return JSONResponse({"success": True, "is_retinal": True, **result})
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(500, f"Prediction failed: {str(e)}")
