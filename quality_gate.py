import json
import sys

THRESHOLD = 0.85

with open("metrics.json") as f:
    acc = json.load(f)["accuracy"]

print(f"Accuracy: {acc:.4f} | Threshold: {THRESHOLD}")
if acc < THRESHOLD:
    print("QUALITY GATE FAILED")
    sys.exit(1)
print("QUALITY GATE PASSED")
