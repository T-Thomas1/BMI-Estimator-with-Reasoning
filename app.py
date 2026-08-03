from src.models.pose_estimator import PoseEstimator
from src.models.bmi_predictor import BMIPredictor

def estimate_bmi(image_path):
    # Extract body ratios
    pose = PoseEstimator()
    ratios = pose.extract_ratios(image_path)

    if not ratios:
        return "No person detected in image"

    # Predict BMI
    predictor = BMIPredictor()
    bmi, category = predictor.predict(ratios.swr, ratios.whr)

    return {
        'bmi': round(bmi, 1),
        'category': category,
        'swr': round(ratios.swr, 3),
        'whr': round(ratios.whr, 3)
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = estimate_bmi(sys.argv[1])
        print(f"\nBMI: {result['bmi']} ({result['category']})")
        print(f"Shoulder/Hip: {result['swr']}")
        print(f"Waist/Height: {result['whr']}")
    else:
        print("Usage: python app.py photo.jpg")