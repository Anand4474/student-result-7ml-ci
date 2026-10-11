import unittest

from app import app


class TestApp(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["status"], "healthy")

    def test_predict(self):
        r = self.client.post("/predict", json={
            "attendance": 90, "internal_marks": 85,
            "assignment_marks": 88, "previous_score": 80})
        self.assertEqual(r.status_code, 200)
        self.assertIn("prediction", r.get_json())

    def test_predict_missing_field(self):
        r = self.client.post("/predict", json={"attendance": 90})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
