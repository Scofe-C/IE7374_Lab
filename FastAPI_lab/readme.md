# Lab 4 — FastAPI: Breast Cancer Classifier

## Overview

This lab exposes a trained **RandomForest** classifier as a REST API using **FastAPI** and **uvicorn**, with a browser-based UI served directly from the API.

- **Dataset**: Breast Cancer Wisconsin (569 samples, 30 features)
- **Task**: Binary classification — `malignant (0)` vs `benign (1)`
- **Classifier**: `RandomForestClassifier` (100 estimators)
- **Test Accuracy**: ~95.6%

**Key differences from the spec example:**

| Aspect | Spec (Iris) | This Lab |
|---|---|---|
| Dataset | Iris (4 features) | Breast Cancer Wisconsin (30 features) |
| Classifier | Decision Tree | Random Forest |
| Response | prediction only | prediction + label + confidence |
| Extra endpoint | none | `/health` — model load status |
| Frontend | none | Browser UI served at `GET /` |

---

## Project Structure

```
FastAPI_lab/
├── model/
│   └── cancer_model.pkl      ← generated after running train.py
├── src/
│   ├── __init__.py
│   ├── data.py               ← dataset loading + train/test split
│   ├── train.py              ← train RF model, save .pkl
│   ├── predict.py            ← load model, run inference
│   └── main.py               ← FastAPI app + all endpoints
├── ui.html                   ← browser UI (served at http://127.0.0.1:8000)
├── README.md
└── requirements.txt
```

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv fastapi_lab_env
fastapi_lab_env\Scripts\activate       # Windows
# source fastapi_lab_env/bin/activate  # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Running the Lab

### Step 1 — Train the model

```bash
cd src
python train.py
```

Expected output:
```
Test Accuracy: 0.9561

Classification Report:
              precision    recall  f1-score   support
   malignant       0.95      0.93      0.94        42
      benign       0.96      0.97      0.97        72
```

This saves `model/cancer_model.pkl`.

### Step 2 — Start the API server

```bash
uvicorn main:app --reload
```

### Step 3 — Open the UI

Navigate to **[http://127.0.0.1:8000](http://127.0.0.1:8000)** — the browser UI loads directly.

- Use **Malignant Sample** or **Benign Sample** buttons to auto-fill real data points
- Or enter your own 30 feature values manually
- Click **Run Inference** to get a prediction

For the raw API docs: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

## Endpoints

### `GET /`
Serves the browser UI (`ui.html`).

### `GET /health`

Returns model load status and classifier name.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "classifier": "RandomForestClassifier"
}
```

### `POST /predict`

Accepts 30 numeric features, returns classification result.

**Request body:**
```json
{
  "mean_radius": 17.99,
  "mean_texture": 10.38,
  "mean_perimeter": 122.8,
  "mean_area": 1001.0,
  "mean_smoothness": 0.1184,
  "mean_compactness": 0.2776,
  "mean_concavity": 0.3001,
  "mean_concave_points": 0.1471,
  "mean_symmetry": 0.2419,
  "mean_fractal_dimension": 0.07871,
  "radius_error": 1.095,
  "texture_error": 0.9053,
  "perimeter_error": 8.589,
  "area_error": 153.4,
  "smoothness_error": 0.006399,
  "compactness_error": 0.04904,
  "concavity_error": 0.05373,
  "concave_points_error": 0.01587,
  "symmetry_error": 0.03003,
  "fractal_dimension_error": 0.006193,
  "worst_radius": 25.38,
  "worst_texture": 17.33,
  "worst_perimeter": 184.6,
  "worst_area": 2019.0,
  "worst_smoothness": 0.1622,
  "worst_compactness": 0.6656,
  "worst_concavity": 0.7119,
  "worst_concave_points": 0.2654,
  "worst_symmetry": 0.4601,
  "worst_fractal_dimension": 0.1189
}
```

**Response:**
```json
{
  "prediction": 0,
  "label": "malignant",
  "confidence": 0.97,
  "model": "RandomForestClassifier"
}
```

**Error responses:**
- `503` — Model not loaded (run `train.py` first)
- `422` — Validation error (missing or wrong-type fields)
- `500` — Internal inference error