# src/models/bmi_predictor.py (updated version)
import joblib
from pathlib import Path
from config.settings import config

class BMIPredictor:
    def __init__(self, use_real_data=True):
        if use_real_data:
            model_path = config.MODELS_DIR / "checkpoints" / "bmi_model_real.pkl"
        else:
            model_path = config.MODELS_DIR / "checkpoints" / "bmi_model.pkl"

        if model_path.exists():
            data = joblib.load(model_path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_trained = True
            print(f"✅ Loaded model from {model_path}")
        else:
            print(f"⚠️ No model found at {model_path}")
            self.is_trained = False

    def predict(self, swr, whr):
        if not self.is_trained:
            raise ValueError("Model not trained. Run train_with_real_data.py first")

        X = self.scaler.transform([[swr, whr]])
        bmi = self.model.predict(X)[0]

        # Categorize
        if bmi < 18.5:
            category = "Underweight"
        elif bmi < 25:
            category = "Normal weight"
        elif bmi < 30:
            category = "Overweight"
        else:
            category = "Obese"

        return bmi, category