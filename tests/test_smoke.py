import os
import joblib

def test_model_file_exists():
    assert os.path.exists("model.pkl"), "model.pkl manquant à la racine du projet"

def test_model_loads():
    # Vérifie simplement que le pickle se charge (version/scikit-learn compatibles)
    joblib.load("model.pkl")
