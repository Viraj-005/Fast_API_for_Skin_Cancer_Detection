import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    print(response.json())  # Print the actual response for debugging
    assert response.status_code == 200
    assert response.json() == {"message": "Skin Cancer Detection Model API"}

def test_predict_skin_cancer():
    try:
        files = {'file': ('test_image.jpg', open('test_image.jpg', 'rb'), 'image/jpeg')}
        response = client.post("/predict", files=files)
        assert response.status_code == 200
        assert "prediction" in response.json()
    except FileNotFoundError:
        pytest.fail("test_image.jpg not found. Please ensure the test image is available.")