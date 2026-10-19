from fastapi.testclient import TestClient

from heart_mLOps import api


def test_prediction_endpoint(monkeypatch):
    class StubModel:
        def predict_proba(self, frame):
            assert list(frame.columns) == api.FEATURES
            return [[0.2, 0.8]]

    monkeypatch.setattr(api, "model", StubModel())
    patient = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
               "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3,
               "slope": 3, "ca": 0, "thal": 6}
    response = TestClient(api.app).post("/predict", json=patient)
    assert response.status_code == 200
    assert response.json()["prediction"] == 1
    assert response.json()["confidence"] == 0.8


def test_prediction_rejects_unknown_fields():
    patient = {"age": 50, "extra": 1}
    response = TestClient(api.app).post("/predict", json=patient)
    assert response.status_code == 422
