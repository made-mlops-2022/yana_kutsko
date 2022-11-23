import pytest
from fastapi.testclient import TestClient
from online_inference.generate_test_data import generate_valid_data, generate_invalid_data
from online_inference.main import app, load_model

client = TestClient(app)


@pytest.fixture(scope='session', autouse=True)
def initialize_model():
    load_model()


def test_predict_for_valid_data():
    data = generate_valid_data()
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert (response.json() == {"result": [0]} or response.json() == {"result": [1]})


def test_predict_for_invalid_data():
    data = generate_invalid_data()
    response = client.post("/predict", json=data)
    assert response.status_code == 400
    assert response.content.decode().startswith("Data is invalid")
    assert "age is beyond allowed boundaries" in response.content.decode()


def test_predict_for_missing_field():
    data = generate_invalid_data()
    data.pop("age")
    response = client.post("/predict", json=data)
    assert response.status_code == 400
    assert response.content.decode().startswith("Data is invalid")
    assert "age\n  field required" in response.content.decode()
