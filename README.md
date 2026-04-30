# OcularAI — Full Deployment Guide
## DenseNet121 + SVM → FastAPI Backend → Vercel Frontend

---

## STEP 1 — Google Drive থেকে File ID বের করো

তোমার Google Drive এ যাও → `FYDP_FullDataset` folder খোলো।

প্রতিটা file এর জন্য:
1. File টার উপর right-click করো
2. **"Share"** → **"Anyone with the link"** করো
3. Link টা copy করো → এইরকম দেখাবে:
   `https://drive.google.com/file/d/1ABC123xyz.../view`
4. মাঝের অংশটাই **File ID**: `1ABC123xyz...`

তোমার দরকার এই 5টা file এর ID:
```
densenet121_finetuned.h5   →  ID: _______________
https://drive.google.com/file/d/1FoVidXa3rt5ANpSPNzUrxwCN1Xv9Wg_M/view?usp=sharing
best_clf.pkl               →  ID: _______________
https://drive.google.com/file/d/1jBE3tYjqG_nWw5Y7KRazD1SvDdHrhHgZ/view?usp=sharing
selector.pkl               →  ID: _______________
https://drive.google.com/file/d/1vFyw9karH1eGttLdfP-N0aK2uoXHOJpf/view?usp=sharing
scaler.pkl                 →  ID: _______________
https://drive.google.com/file/d/156xcXLRkd4h2CWvBPIjZQ-MYu4RoJS3R/view?usp=sharing
pca.pkl                    →  ID: _______________
https://drive.google.com/file/d/1fNf7Hp9Yk7yuooqohkiZ5sAmTB_8ut2e/view?usp=sharing
```
https://drive.google.com/file/d/1FoVidXa3rt5ANpSPNzUrxwCN1Xv9Wg_M/view?usp=sharing
https://drive.google.com/file/d/1jBE3tYjqG_nWw5Y7KRazD1SvDdHrhHgZ/view?usp=sharing
https://drive.google.com/file/d/1vFyw9karH1eGttLdfP-N0aK2uoXHOJpf/view?usp=drive_link
https://drive.google.com/file/d/156xcXLRkd4h2CWvBPIjZQ-MYu4RoJS3R/view?usp=drive_link
https://drive.google.com/file/d/1fNf7Hp9Yk7yuooqohkiZ5sAmTB_8ut2e/view?usp=drive_link
---

## STEP 2 — main.py তে File ID বসাও

`main.py` ফাইলে এই অংশটা খোঁজো:

```python
GDRIVE_FILES = {
    "densenet121_finetuned.h5": "YOUR_H5_FILE_ID_HERE",
    "best_clf.pkl":             "YOUR_CLF_FILE_ID_HERE",
    "selector.pkl":             "YOUR_SELECTOR_FILE_ID_HERE",
    "scaler.pkl":               "YOUR_SCALER_FILE_ID_HERE",
    "pca.pkl":                  "YOUR_PCA_FILE_ID_HERE",
}
```

প্রতিটা `"YOUR_..._HERE"` এর জায়গায় তোমার actual File ID বসাও।

---

## STEP 3 — GitHub এ Push করো

```bash
# Terminal/CMD এ:
cd ocularai-backend
git init
git add .
git commit -m "OcularAI backend initial"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ocularai-backend.git
git push -u origin main
```

---

## STEP 4 — Render.com এ Backend Deploy করো (FREE)

1. যাও → https://render.com → Sign up (GitHub দিয়ে)
2. **"New +"** → **"Web Service"**
3. তোমার GitHub repo select করো
4. Settings:
   - **Name:** `ocularai-backend`
   - **Region:** `Singapore` (Bangladesh এর কাছে)
   - **Branch:** `main`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free
5. **"Create Web Service"** চাপো
6. Deploy হতে 3-5 মিনিট লাগবে
7. URL পাবে এইরকম: `https://ocularai-backend.onrender.com`

⚠ প্রথমবার request এ model download হবে (1-2 মিনিট)।
পরের request থেকে fast হবে কারণ /tmp তে cached থাকবে।

---

## STEP 5 — Frontend এ Backend URL বসাও

`frontend/index.html` এ এই line খোঁজো:

```javascript
const BACKEND_URL = "https://YOUR-APP-NAME.onrender.com";
```

তোমার Render URL দিয়ে replace করো:

```javascript
const BACKEND_URL = "https://ocularai-backend.onrender.com";
```

---

## STEP 6 — Vercel এ Frontend Deploy করো

1. যাও → https://vercel.com → Sign up
2. **"Add New Project"**
3. `frontend` folder টা drag & drop করো
   অথবা GitHub repo থেকে import করো
4. **Deploy** চাপো
5. URL পাবে: `https://ocularai.vercel.app`

---

## Final Architecture

```
User (Browser)
    │
    │  Image upload
    ▼
Vercel (index.html)
    │
    │  POST /predict  (FormData with image)
    ▼
Render.com FastAPI Server
    │
    ├── Download models from Google Drive (first time only)
    │
    ├── DenseNet121 → feature extract (1024-dim vector)
    ├── StandardScaler → normalize
    ├── SelectKBest → top features
    └── SVM.predict() → disease class + probability
    │
    │  JSON response
    ▼
Vercel (shows result card)
```

---

## Test করো

Backend deploy হওয়ার পর এই URL এ যাও:
`https://YOUR-APP.onrender.com/health`

দেখাবে: `{"status":"healthy","model_loaded":false}`

Predict করার পর: `{"status":"healthy","model_loaded":true}`

---

## সমস্যা হলে

| সমস্যা | সমাধান |
|--------|---------|
| `403 Forbidden` Google Drive থেকে | File sharing "Anyone with link" করো |
| `CORS error` browser এ | main.py তে `allow_origins=["*"]` আছে কিনা দেখো |
| First request timeout | Render free tier cold start 30-60s নেয় — স্বাভাবিক |
| Model load error | .h5 file টা সঠিকভাবে save হয়েছিল কিনা দেখো |
