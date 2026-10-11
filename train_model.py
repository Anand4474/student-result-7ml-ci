import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

FEATURES = ["attendance", "internal_marks", "assignment_marks", "previous_score"]


def make_data(n=600, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(30, 100, size=(n, 4))
    score = 0.2 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2] + 0.3 * X[:, 3]
    y = (score >= 60).astype(int)
    return X, y


def main():
    X, y = make_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    joblib.dump(model, "student_result_model.pkl")
    with open("metrics.json", "w") as f:
        json.dump({"accuracy": acc}, f)
    print(f"Model trained. Accuracy = {acc:.4f}")


if __name__ == "__main__":
    main()
