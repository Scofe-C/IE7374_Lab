from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
import os

from predict import predict, is_model_loaded, get_model

app = FastAPI(
    title="Breast Cancer Classifier API",
    description="Serves a RandomForest model trained on the Breast Cancer Wisconsin dataset.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to ui.html — one level up from src/
UI_PATH = os.path.join(os.path.dirname(__file__), "..", "ui.html")


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class CancerData(BaseModel):
    """30 numeric features from the Breast Cancer Wisconsin dataset."""
    mean_radius: float = Field(..., example=17.99)
    mean_texture: float = Field(..., example=10.38)
    mean_perimeter: float = Field(..., example=122.8)
    mean_area: float = Field(..., example=1001.0)
    mean_smoothness: float = Field(..., example=0.1184)
    mean_compactness: float = Field(..., example=0.2776)
    mean_concavity: float = Field(..., example=0.3001)
    mean_concave_points: float = Field(..., example=0.1471)
    mean_symmetry: float = Field(..., example=0.2419)
    mean_fractal_dimension: float = Field(..., example=0.07871)
    radius_error: float = Field(..., example=1.095)
    texture_error: float = Field(..., example=0.9053)
    perimeter_error: float = Field(..., example=8.589)
    area_error: float = Field(..., example=153.4)
    smoothness_error: float = Field(..., example=0.006399)
    compactness_error: float = Field(..., example=0.04904)
    concavity_error: float = Field(..., example=0.05373)
    concave_points_error: float = Field(..., example=0.01587)
    symmetry_error: float = Field(..., example=0.03003)
    fractal_dimension_error: float = Field(..., example=0.006193)
    worst_radius: float = Field(..., example=25.38)
    worst_texture: float = Field(..., example=17.33)
    worst_perimeter: float = Field(..., example=184.6)
    worst_area: float = Field(..., example=2019.0)
    worst_smoothness: float = Field(..., example=0.1622)
    worst_compactness: float = Field(..., example=0.6656)
    worst_concavity: float = Field(..., example=0.7119)
    worst_concave_points: float = Field(..., example=0.2654)
    worst_symmetry: float = Field(..., example=0.4601)
    worst_fractal_dimension: float = Field(..., example=0.1189)


class CancerResponse(BaseModel):
    """Prediction result with label, confidence, and model metadata."""
    prediction: int = Field(..., description="0 = malignant, 1 = benign")
    label: str = Field(..., description="Human-readable class name")
    confidence: float = Field(..., description="Model's confidence (max class probability)")
    model: str = Field(..., description="Classifier class name")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classifier: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["UI"])
async def serve_ui():
    """Serve the frontend UI."""
    ui_path = os.path.abspath(UI_PATH)
    if not os.path.exists(ui_path):
        raise HTTPException(status_code=404, detail="ui.html not found. Make sure it is in the FastAPI_lab/ folder.")
    return FileResponse(ui_path, media_type="text/html")


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    """Check whether the API is running and the model is loaded."""
    loaded = is_model_loaded()
    classifier = type(get_model()).__name__ if loaded else None
    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        classifier=classifier,
    )


@app.post("/predict", response_model=CancerResponse, tags=["Inference"])
async def predict_cancer(data: CancerData):
    """
    Classify a tumor as malignant or benign.

    Accepts 30 numeric features matching the Breast Cancer Wisconsin dataset schema.
    Returns the predicted class, human-readable label, confidence score, and model name.
    """
    features = list(data.model_dump().values())

    try:
        result = predict(features)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return CancerResponse(**result)