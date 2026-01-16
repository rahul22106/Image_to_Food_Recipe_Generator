import os
import sys
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
from typing import List, Optional
from pydantic import BaseModel
from pathlib import Path

from Recipe_Generator.pipeline.prediction_pipeline import PredictionPipeline
from Recipe_Generator.logger import logger
from Recipe_Generator.exception import CustomException


app = FastAPI(
    title="Recipe Generator API",
    description="AI-powered food image to recipe prediction API",
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


class RecipeResponse(BaseModel):
    rank: int
    name: str
    similarity_score: float
    ingredients: str
    instructions: str


class PredictionResponse(BaseModel):
    success: bool
    message: str
    predictions: List[RecipeResponse]
    total_predictions: int


class HealthResponse(BaseModel):
    status: str
    message: str


pipeline = None


@app.on_event("startup")
async def startup_event():
    global pipeline
    try:
        logger.info("Loading prediction pipeline...")
        pipeline = PredictionPipeline()
        logger.info("Pipeline loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load pipeline: {str(e)}")
        raise


@app.get("/", response_model=HealthResponse)
async def root():
    return {
        "status": "running",
        "message": "Recipe Generator API is running. Visit /docs for API documentation."
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    
    return {
        "status": "healthy",
        "message": "API is healthy and ready to accept requests"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_recipe(
    file: UploadFile = File(...),
    top_k: Optional[int] = 5
):
    try:
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not loaded")
        
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        if top_k < 1 or top_k > 20:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")
        
        logger.info(f"Received prediction request: {file.filename}, top_k={top_k}")
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        predictions = pipeline.predict_from_image_object(image, top_k=top_k)
        
        recipe_responses = []
        for idx, pred in enumerate(predictions, 1):
            recipe_responses.append(
                RecipeResponse(
                    rank=idx,
                    name=pred.get('name', 'Unknown'),
                    similarity_score=round(pred.get('similarity_score', 0.0), 4),
                    ingredients=pred.get('ingredients', 'Not available'),
                    instructions=pred.get('instructions', 'Not available')
                )
            )
        
        logger.info(f"Prediction successful: {len(recipe_responses)} recipes returned")
        
        return PredictionResponse(
            success=True,
            message="Prediction completed successfully",
            predictions=recipe_responses,
            total_predictions=len(recipe_responses)
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    top_k: Optional[int] = 5
):
    try:
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not loaded")
        
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 images allowed per batch")
        
        if top_k < 1 or top_k > 20:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")
        
        results = []
        
        for file in files:
            if not file.content_type.startswith("image/"):
                continue
            
            try:
                contents = await file.read()
                image = Image.open(io.BytesIO(contents)).convert('RGB')
                
                predictions = pipeline.predict_from_image_object(image, top_k=top_k)
                
                recipe_responses = []
                for idx, pred in enumerate(predictions, 1):
                    recipe_responses.append({
                        "rank": idx,
                        "name": pred.get('name', 'Unknown'),
                        "similarity_score": round(pred.get('similarity_score', 0.0), 4),
                        "ingredients": pred.get('ingredients', 'Not available'),
                        "instructions": pred.get('instructions', 'Not available')
                    })
                
                results.append({
                    "filename": file.filename,
                    "success": True,
                    "predictions": recipe_responses
                })
                
            except Exception as e:
                logger.error(f"Failed to process {file.filename}: {str(e)}")
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
        
        return JSONResponse(content={
            "success": True,
            "message": f"Processed {len(results)} images",
            "results": results
        })
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/info")
async def api_info():
    return {
        "api_name": "Recipe Generator API",
        "version": "1.0.0",
        "description": "AI-powered food image to recipe prediction",
        "endpoints": {
            "/": "API root and health check",
            "/health": "Health check endpoint",
            "/predict": "Single image prediction",
            "/predict/batch": "Batch image prediction",
            "/docs": "Interactive API documentation",
            "/redoc": "Alternative API documentation"
        },
        "model_info": {
            "vision_model": "ResNet50",
            "text_model": "sentence-transformers/all-MiniLM-L6-v2",
            "fusion_model": "Multimodal Contrastive Learning"
        }
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("STARTING RECIPE GENERATOR API")
    print("="*70)
    print("\n🚀 API Server starting...")
    print("\n📍 Endpoints:")
    print("   - Health Check: http://localhost:8000/health")
    print("   - Predict: http://localhost:8000/predict")
    print("   - API Docs: http://localhost:8000/docs")
    print("   - ReDoc: http://localhost:8000/redoc")
    print("\n" + "="*70 + "\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )