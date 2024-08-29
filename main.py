from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import io
import logging
from mangum import Mangum

app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load your trained model
try:
    model = load_model('skin_cancer_model.h5')
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise RuntimeError("Model loading failed.") from e

class PredictionResponse(BaseModel):
    prediction: str
    probability: float

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess the image for model prediction.

    Args:
    image: PIL Image object

    Returns:
    np.ndarray: Preprocessed image data
    """
    IMAGE_SIZE = 224  # Define the image size as a constant
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))  # Resize to the input size your model expects
    image = np.array(image) / 255.0  # Normalize to [0, 1] range
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.get('/')
def read_root():
    return {'message': 'Skin Cancer Detection Model API'}

handler = Mangum(app=app)

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # Ensure the file is not empty
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Validate that the image is in a supported format
        if image.format not in ['JPEG', 'PNG', 'JPG']:
            raise HTTPException(status_code=400, detail="Unsupported image format. Please upload a JPEG or PNG image.")

        input_data = preprocess_image(image)
        
        # Make the prediction
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction, axis=1)[0]
        probability = np.max(prediction)

        # Log the prediction array and the predicted class
        logger.info(f"Prediction array: {prediction}")
        logger.info(f"Predicted class: {predicted_class}")

        # Map predicted_class to human-readable label
        label = "Non-cancerous" if predicted_class == 0 else "Cancerous"

        # Optional: Apply a threshold to make the classification more robust
        threshold = 0.5  # Adjust the threshold if necessary
        if probability < threshold:
            label = "Non-cancerous"
            probability = 1 - probability

        logger.info(f"Final Prediction: {label} with probability {probability:.2f}")
        return PredictionResponse(prediction=label, probability=float(probability))
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail="Failed to process file")
