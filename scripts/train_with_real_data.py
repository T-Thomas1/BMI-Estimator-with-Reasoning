# scripts/train_with_real_data.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load_real_data import combine_datasets, prepare_training_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from config.settings import config

def main():
    print("=" * 60)
    print("TRAINING BMI MODEL WITH BEDLAM DATA")
    print("=" * 60)

    # Load datasets (BEDLAM + UniqueData when available)
    df = combine_datasets()

    if df is None or len(df) == 0:
        print("❌ No data loaded! Check your BEDLAM path.")
        return

    # Prepare training data
    X_train, X_test, y_train, y_test, scaler = prepare_training_data(df)

    # Train model
    print("\n🚀 Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n✅ Model Performance:")
    print(f"   MAE: {mae:.2f} BMI points")
    print(f"   R² Score: {r2:.3f}")

    # Save model
    model_path = config.MODELS_DIR / "checkpoints" / "bmi_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        'model': model,
        'scaler': scaler,
        'features': ['swr', 'whr'],
        'performance': {'mae': mae, 'r2': r2}
    }, model_path)

    print(f"\n💾 Model saved to: {model_path}")

if __name__ == "__main__":
    main()