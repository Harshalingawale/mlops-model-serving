#  MLOps Model Serving

> End-to-end machine-learning service: **train → evaluate → version → serve → containerize → test in CI**. A clean reference for taking a model from notebook to production API.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-Serving-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white" />
  <img src="https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

##  What this demonstrates

Most ML tutorials stop at `model.fit()`. This repo shows the part employers actually care about: **operationalizing** a model. It trains a classifier inside a reproducible `Pipeline`, serializes it with metrics and feature contract, serves predictions behind a typed FastAPI, ships in Docker, and is verified by an automated test suite running in GitHub Actions.

> Verified results on the built-in dataset: **ROC-AUC 0.994**, 5-fold CV AUC 0.992 ± 0.006, F1 0.958.

##  Pipeline

```
 src/train.py                         src/serve.py
┌──────────────────────┐            ┌────────────────────────┐
│ load data            │            │ load model.joblib      │
│ StandardScaler       │  artifacts │ /health  /metrics      │
│ RandomForest         │ ─────────▶ │ /features /predict     │
│ CV + holdout eval    │  model +   │ typed Pydantic I/O     │
│ dump model + metrics │  metrics   │ feature-contract check │
└──────────────────────┘            └────────────────────────┘
        │                                      │
        └────────────── pytest + GitHub Actions CI ───────────┘
```

##  Quickstart

```bash
git clone https://github.com/harshalingawale/mlops-model-serving.git
cd mlops-model-serving
make install          # pip install -r requirements.txt
make train            # trains + writes artifacts/{model.joblib,metrics.json,features.json}
make serve            # uvicorn src.serve:app --reload  →  http://127.0.0.1:8000/docs
```

### Get a prediction

```bash
# inspect the expected feature order
curl localhost:8000/features

# request a prediction (30-dim vector here)
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [14.1, 20.2, 92.0, 654.0, 0.1, 0.1, 0.08, 0.05, 0.18, 0.06,
                    0.4, 1.2, 2.8, 40.0, 0.007, 0.02, 0.03, 0.01, 0.02, 0.003,
                    16.3, 27.0, 108.0, 858.0, 0.14, 0.22, 0.27, 0.11, 0.29, 0.08]}'
# → {"prediction": 0, "probability": 0.12}
```

##  Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + whether a model is loaded |
| `GET` | `/metrics` | Training/evaluation metrics |
| `GET` | `/features` | Feature names + count (the input contract) |
| `POST`| `/predict` | Class prediction + probability |

##  Docker

```bash
make docker     # builds the image, trains inside it, serves on :8000
```

##  Tests & CI

```bash
make test
```

Every push runs `python -m src.train` then `pytest` on GitHub Actions, asserting the model still clears an AUC threshold — a lightweight **model-quality gate**.

##  Structure

```
mlops-model-serving/
├── src/
│   ├── train.py     # train, evaluate (CV + holdout), persist model+metrics
│   └── serve.py     # FastAPI app with typed I/O + feature-contract validation
├── tests/test_api.py
├── artifacts/       # generated: model.joblib, metrics.json, features.json
├── Dockerfile · Makefile · requirements.txt
└── .github/workflows/ci.yml
```

##  Use your own data

Swap `load_data()` in `src/train.py` to read your CSV (`pd.read_csv(...)`, return `X, y, feature_names`). Everything downstream — serving, contract checks, CI — keeps working.

##  Tech Stack

**Python · scikit-learn · FastAPI · Pydantic · joblib · Docker · GitHub Actions · pytest**

##  Roadmap

- [ ] MLflow experiment tracking + model registry
- [ ] Prometheus metrics + Grafana dashboard
- [ ] Data/prediction drift monitoring
- [ ] Blue-green deploy via GitHub Actions

##  License

MIT © [Harshal Ingawale](https://github.com/harshalingawale)
