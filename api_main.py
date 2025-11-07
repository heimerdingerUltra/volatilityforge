from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import time
import logging
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import tempfile

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.registry import ModelRegistry
from src.data.advanced_features import AdvancedFeatures
from src.inference.pipeline import InferencePipeline, ModelServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter('volatility_requests_total', 'Total requests', ['endpoint', 'status'])
REQUEST_LATENCY = Histogram('volatility_request_duration_seconds', 'Request latency', ['endpoint'])
PREDICTION_COUNT = Counter('volatility_predictions_total', 'Total predictions made')

app = FastAPI(
    title="Volatility Forge API",
    description="Advanced Implied Volatility Prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_server = None
feature_extractor = None

class PredictionRequest(BaseModel):
    """Single prediction request"""
    BID: float = Field(..., description="Bid price")
    ASK: float = Field(..., description="Ask price")
    BIDSIZE: Optional[float] = Field(None, description="Bid size")
    ASKSIZE: Optional[float] = Field(None, description="Ask size")
    STRIKE_PRC: float = Field(..., description="Strike price")
    DAYS_TO_EXPIRY_CALC: float = Field(..., description="Days to expiry")
    ACVOL_1: Optional[float] = Field(None, description="Volume")
    OPINT_1: Optional[float] = Field(None, description="Open interest")
    OPTION_TYPE: str = Field(..., description="CALL or PUT")
    
    class Config:
        schema_extra = {
            "example": {
                "BID": 10.5,
                "ASK": 11.0,
                "BIDSIZE": 100,
                "ASKSIZE": 150,
                "STRIKE_PRC": 100,
                "DAYS_TO_EXPIRY_CALC": 30,
                "ACVOL_1": 1000,
                "OPINT_1": 5000,
                "OPTION_TYPE": "CALL"
            }
        }

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    data: List[PredictionRequest]

class PredictionResponse(BaseModel):
    """Prediction response"""
    predicted_iv: float
    latency_ms: float
    model_version: str
    uncertainty: Optional[float] = None

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[float]
    latency_ms: float
    count: int
    model_version: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str

class ModelInfo(BaseModel):
    """Model information"""
    name: str
    version: str
    n_features: int
    metrics: Dict


@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup"""
    global model_server, feature_extractor
    
    try:
        logger.info("Initializing Volatility Forge API...")
        
        registry_path = Path("model_registry")
        if not registry_path.exists():
            logger.warning(f"Model registry not found at {registry_path}")
            model_server = None
        else:
            model_server = ModelServer(str(registry_path))
            logger.info("Model server initialized")
        
        feature_extractor = AdvancedFeatures(cache_dir=".cache/api")
        
        logger.info("API initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Volatility Forge API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    REQUEST_COUNT.labels(endpoint='health', status='success').inc()
    
    return HealthResponse(
        status="healthy" if model_server else "degraded",
        model_loaded=model_server is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(request: PredictionRequest):
    """
    Predict implied volatility for a single option
    """
    start_time = time.time()
    
    try:
        if model_server is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        df = pd.DataFrame([request.dict()])
        
        result = model_server.predict("tabpfn", df)
        
        latency_ms = (time.time() - start_time) * 1000
        
        REQUEST_COUNT.labels(endpoint='predict', status='success').inc()
        REQUEST_LATENCY.labels(endpoint='predict').observe(time.time() - start_time)
        PREDICTION_COUNT.inc()
        
        return PredictionResponse(
            predicted_iv=float(result.prediction[0]),
            latency_ms=latency_ms,
            model_version=result.model_version,
            uncertainty=float(result.uncertainty[0]) if result.uncertainty is not None else None
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(endpoint='predict', status='error').inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict implied volatility for multiple options
    """
    start_time = time.time()
    
    try:
        if model_server is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        data_dicts = [r.dict() for r in request.data]
        df = pd.DataFrame(data_dicts)
        
        result = model_server.predict("tabpfn", df)
        
        latency_ms = (time.time() - start_time) * 1000
        
        REQUEST_COUNT.labels(endpoint='predict_batch', status='success').inc()
        REQUEST_LATENCY.labels(endpoint='predict_batch').observe(time.time() - start_time)
        PREDICTION_COUNT.inc(len(request.data))
        
        return BatchPredictionResponse(
            predictions=result.prediction.tolist(),
            latency_ms=latency_ms,
            count=len(request.data),
            model_version=result.model_version
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(endpoint='predict_batch', status='error').inc()
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/file", tags=["Prediction"])
async def predict_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Predict implied volatility from uploaded Excel/CSV file
    """
    start_time = time.time()
    
    try:
        if model_server is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        if file.filename.endswith('.xlsx'):
            df = pd.read_excel(tmp_path)
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(tmp_path)
        else:
            raise HTTPException(status_code=400, detail="Only .xlsx and .csv files supported")
        
        result = model_server.predict("tabpfn", df)
        
        df['predicted_iv'] = result.prediction
        
        output_path = Path(tmp_path).with_suffix('.output.xlsx')
        df.to_excel(output_path, index=False)
        
        latency_ms = (time.time() - start_time) * 1000
        
        REQUEST_COUNT.labels(endpoint='predict_file', status='success').inc()
        PREDICTION_COUNT.inc(len(df))
        
        # Clean up temp files in background
        if background_tasks:
            background_tasks.add_task(lambda: Path(tmp_path).unlink(missing_ok=True))
        
        return FileResponse(
            output_path,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            filename=f"predictions_{Path(file.filename).stem}.xlsx"
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(endpoint='predict_file', status='error').inc()
        logger.error(f"File prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", tags=["Models"])
async def list_models():
    """List available models"""
    try:
        if model_server is None:
            return {"models": []}
        
        registry = ModelRegistry("model_registry")
        models = registry.list_models()
        
        return {"models": models}
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_name}", response_model=ModelInfo, tags=["Models"])
async def get_model_info(model_name: str, version: str = "latest"):
    """Get information about a specific model"""
    try:
        if model_server is None:
            raise HTTPException(status_code=503, detail="Model registry not available")
        
        registry = ModelRegistry("model_registry")
        metadata = registry.get_model_info(model_name, version)
        
        return ModelInfo(
            name=metadata.model_name,
            version=metadata.version,
            n_features=metadata.n_features,
            metrics=metadata.metrics
        )
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def serve():
    """Entry point for running the server"""
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info"
    )


if __name__ == "__main__":
    serve()