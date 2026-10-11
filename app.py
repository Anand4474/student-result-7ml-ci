import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("student_result_model.pkl")
FEATURES = ["attendance", "internal_marks", "assignment_marks", "previous_score"]


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    try:
        row = [[float(data[f]) for f in FEATURES]]
    except (TypeError, ValueError):
        return jsonify({"error": "All fields must be numeric"}), 400
    result = "PASS" if int(model.predict(row)[0]) == 1 else "FAIL"
    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
