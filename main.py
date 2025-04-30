from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

app = FastAPI(title="Chiremba Skin Disease Service",
             description="API for skin disease detection and classification",
             version="1.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5000",
        "https://chiremba-ai-frontend-production.up.railway.app",
        "https://chiremba-full-stack-160376271578.us-central1.run.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Updated class labels for Skinalyze
SKINALYZE_CLASS_LABELS = [
    'Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox'
]

skinalyze_model = None

def load_model(model_path: str):
    """Load the TensorFlow model from the specified path."""
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint required for Cloud Run"""
    return {"status": "healthy"}

@app.get("/")
async def root():
    """Root endpoint providing API information"""
    return {
        "message": "Chiremba Skin Disease Detection API",
        "version": "1.0.0",
        "status": "ok"
    }

@app.post("/test")
async def test_endpoint(file: UploadFile = File(...)):
    """Test endpoint for connectivity checks"""
    try:
        return JSONResponse(content={
            "predicted_class": "test_success",
            "confidence": 1.0,
            "message": f"File received successfully: {file.filename}"
        })
    except Exception as e:
        logger.error(f"Error in test endpoint: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.post("/skindisease_classification")
async def skindisease_classification(file: UploadFile = File(...)):
    """
    Endpoint for skin disease classification
    Returns predictions with confidence scores for the top 3 most likely conditions
    """
    global skinalyze_model
    try:
        logger.info(f"Received skin disease classification request: {file.filename}")
        if skinalyze_model is None:
            skinalyze_model = load_model("SkinNet-23M.h5")
        
        image_data = file.file.read()
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess the image
        image = image.resize((180, 180))
        image_array = tf.keras.utils.img_to_array(image)
        image_array = tf.expand_dims(image_array, 0)
        
        # Make prediction
        prediction = skinalyze_model.predict(image_array)
        probabilities = tf.nn.softmax(prediction[0]).numpy()
        
        # Get top 3 predictions
        sorted_indices = np.argsort(probabilities)[::-1]
        top_3_indices = sorted_indices[:3]
        top_3_classes = [SKINALYZE_CLASS_LABELS[i] for i in top_3_indices]
        top_3_confidences = [float(probabilities[i]) for i in top_3_indices]
        
        logger.info(f"Top 3 predictions: {list(zip(top_3_classes, top_3_confidences))}")
        
        return JSONResponse(content={
            "predicted_class": str(top_3_classes[0]),
            "confidence": top_3_confidences[0],
            "model_used": "SkinNet-23M",
            "alternatives": [
                {"class": top_3_classes[1], "confidence": top_3_confidences[1]},
                {"class": top_3_classes[2], "confidence": top_3_confidences[2]}
            ]
        })
    except Exception as e:
        logger.error(f"Error in skin disease classification: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.middleware("http")
async def log_requests(request, call_next):
    """Middleware to log all HTTP requests and responses"""
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response

if __name__ == "__main__":
    import uvicorn
    
    load_dotenv()
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
