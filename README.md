# Heart Disease MLOps Assignment

An end-to-end, reproducible classifier and inference API built from the supplied UCI Heart Disease data. This is an educational project: output is not a diagnosis or medical advice.

## Data and acquisition

The provided files under `heart+disease/` are the processed 14-column UCI tables for Cleveland, Hungary, Switzerland, and VA Long Beach. The official source describes the four databases and the original target scale (0-4): [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart%2Bdisease). Preserve the supplied raw files and run the included preparation script to produce the cleaned, combined CSV:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python scripts/prepare_data.py
```

The resulting `data/processed/heart_disease.csv` has 13 input features, a site label for descriptive analysis, and a binary target where raw values greater than zero mean disease is present. Missing `?` and `-9` values become missing values and are imputed inside the training pipeline. Site is deliberately excluded from model inputs. The four sites are pooled for this assignment; this is not a prospective or external validation.

## EDA, models, and tracking

```powershell
python scripts/eda.py
python -m heart_mLOps.train
mlflow ui --backend-store-uri .\mlruns
```

EDA plots and an EDA summary are written under `artifacts/eda/`. Training uses a stratified 80/20 split (seed 42), then five-fold stratified cross-validation on the training partition. Logistic Regression and Random Forest share median/most-frequent imputation; numeric fields are standardized and categorical fields are one-hot encoded. Accuracy, precision, recall, and ROC-AUC are reported. Candidate selection uses mean CV recall (to reduce missed positive examples), with CV ROC-AUC as tie-breaker. Test metrics remain a final holdout check. MLflow logs parameters, CV/test metrics, ROC and classification report artifacts, and each fitted pipeline. The selected serializable pipeline is `artifacts/heart_disease_model.joblib`.

## API and local Docker

The required input keys are `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, and `thal`. Start the API:

```powershell
docker compose up --build
```

Try `GET http://localhost:8000/health`, `GET http://localhost:8000/metrics`, or `POST http://localhost:8000/predict` with this JSON:

```json
{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}
```

The response includes a binary prediction and class confidence. Interactive schema docs are at `http://localhost:8000/docs`. Requests and latency are logged; `/metrics` exposes Prometheus counters and model availability.

## CI, tests, monitoring, and deployment

GitHub Actions in `.github/workflows/ci.yml` installs pinned dependencies, lints, prepares data, produces EDA, runs tests, trains the models, and uploads MLflow and model artifacts. Locally, use `ruff check src scripts tests` and `pytest -q`.

For Prometheus and Grafana, run `docker compose -f deploy/monitoring-compose.yaml up --build`; Prometheus is available on port 9090 and Grafana on 3000 (local demo credentials: `admin` / `assignment-local-only`). For Docker Desktop Kubernetes, build the image locally first, then apply `deploy/kubernetes/deployment.yaml`. With Minikube, load the locally built image with `minikube image load heart-disease-api:local` before applying the manifest. Check rollout and service access with `kubectl rollout status deployment/heart-disease-api`, `kubectl get pods,svc`, and `kubectl port-forward service/heart-disease-api 8080:80`. A cloud LoadBalancer requires a configured cloud cluster/provider.

## Assignment deliverables and limitations

The repository includes source data, preparation and EDA scripts, training/inference code, tests, requirements, Docker and Compose files, CI workflow, Kubernetes manifests, monitoring configuration, and `reports/assignment_report.md`. The final report also needs actual run metrics, screenshots of the CI/deployment, a repository URL, and a short recording. These require running the workflow and publishing/recording from the student's own GitHub and local cluster. No public endpoint or screenshots are claimed by this source tree. Complete `reports/evidence_checklist.md` after running locally.

## Dataset citation

UCI Machine Learning Repository. *Heart Disease* dataset, DOI: [10.24432/C52P4X](https://doi.org/10.24432/C52P4X). The original data notes request acknowledging the institutional investigators listed in `heart+disease/heart-disease.names` if publishing results.

