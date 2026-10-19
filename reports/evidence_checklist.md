# Evidence checklist before submission

- [ ] Run the full clean setup on the intended machine and keep the terminal logs.
- [ ] Capture EDA figures from `artifacts/eda/` and document actual metrics from `artifacts/training_summary.json`.
- [ ] Open MLflow UI and capture the comparison of both runs.
- [ ] Build and start the Docker image, call `/health` and `/predict`, and save terminal/browser screenshots.
- [ ] Apply Kubernetes manifests, verify ready replicas and the service endpoint, and capture the cluster/deployment view.
- [ ] Start monitoring compose, send requests, and capture Prometheus and Grafana.
- [ ] Record a short video walking through data prep, CI, training, image/API, deployment, and monitoring.
- [ ] Push to the student's GitHub repository and add its URL to the report.
- [ ] Review data limitations, ethical caveats, and the course policy; make sure you can explain each implementation choice.
