import pytest
from pydantic import ValidationError

from app.main import predict_endpoint, PredictionRequest

@pytest.mark.anyio
def test_predict_success_features_vide():
    data = PredictionRequest(features=[])
    resp = predict_endpoint(data)
    assert resp == {"predictions": []}

@pytest.mark.anyio
def test_predict_success_features_negatif():
    data = PredictionRequest(features=[-2.0, -4.5, -10.0])
    resp = predict_endpoint(data)
    assert resp == {"predictions": [-4.0, -9.0, -20.0]}

@pytest.mark.anyio
def test_predict_success_basic():
    data = PredictionRequest(features=[3.5, 1.2, 4.9])
    resp = predict_endpoint(data)
    assert resp == {"predictions": [7.0, 2.4, 9.8]}


@pytest.mark.anyio
def test_predict_success_string():
    data = PredictionRequest(features=["3.5", "1.2", "4.9"])
    resp = predict_endpoint(data)
    assert resp == {"predictions": [7.0, 2.4, 9.8]}

@pytest.mark.anyio
def test_predict_invalid_data ():
    with pytest.raises(ValidationError):
        PredictionRequest(features=["a", "b", "c"])