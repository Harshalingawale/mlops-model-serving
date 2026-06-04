import json
from pathlib import Path

from fastapi.testclient import TestClient

from src import train, serve

ART = Path(__file__).resolve().parent.parent / "artifacts"


def setup_module(_):
    if not (ART / "model.joblib").exists():
        train.main()
    serve._load()


def test_health():
    client = TestClient(serve.app)
    assert client.get("/health").json()["status"] == "ok"


def test_predict():
    client = TestClient(serve.app)
    feats = json.loads((ART / "features.json").read_text())
    payload = {"features": [0.0] * len(feats)}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json()["prediction"] in (0, 1)


def test_metrics_have_auc():
    m = train.main()
    assert m["roc_auc"] > 0.9
