import os
import unittest

import joblib


class TestMLPipeline(unittest.TestCase):
    def test_model_file_exists(self):
        self.assertTrue(os.path.exists("student_result_model.pkl"))

    def test_model_predicts(self):
        model = joblib.load("student_result_model.pkl")
        self.assertEqual(int(model.predict([[95, 90, 92, 90]])[0]), 1)
        self.assertEqual(int(model.predict([[35, 35, 35, 35]])[0]), 0)


if __name__ == "__main__":
    unittest.main()
