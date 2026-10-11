# Student Result Prediction - ML CI/CD (Practical 7)

GitHub Actions workflow: train model -> quality gate -> unit tests -> Docker build ->
run container -> /health check -> /predict check -> logs + cleanup.

Local run:
    pip install -r requirements.txt
    python train_model.py && python quality_gate.py
    python -m unittest discover -v
    docker build -t student-result-ml . && docker run -p 5000:5000 student-result-ml
