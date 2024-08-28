import pytest
import warnings
from fastapi.testclient import TestClient
from main import app
import os

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Skin Cancer Detection Model API"}

def test_predict_skin_cancer():
    test_image_path = 'test_image.jpg'
    
    # Check if the test image exists
    if not os.path.isfile(test_image_path):
        pytest.fail(f"{test_image_path} not found. Please ensure the test image is available.")

    try:
        with open(test_image_path, 'rb') as img_file:
            files = {'file': ('test_image.jpg', img_file, 'image/jpeg')}
            response = client.post("/predict", files=files)
            assert response.status_code == 200
            assert "prediction" in response.json()
    except OSError as e:
        pytest.fail(f"Error opening file: {e}")
