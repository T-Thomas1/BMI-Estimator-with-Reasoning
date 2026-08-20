# BMI Estimator with Reasoning

A ML system that takes DigitalScale's DenseNet architecture and implements it on a synthetic BEDLAM dataset. This project validates the model's predictions for BMI with a natural language reasoning.

## Problem Statement

Recently tools available during telehealth, or fitness tracking have been readily available. DigitalScale provides photo BMI estimation. This project investigates the final model performance through training, calibration, and evaluation. Conformal Prediction allows a statistical guarantee of surity that the model's prediction is correct. Paired with GPT-as-Judge provides an explainable validation pipeline.

## How It Works

The system operates in three stages:

1. Train The model
2. Run The model on calibrated data
3. Calculate absolute non-conformity
4. Find Q_Hat threshold and apply to our new data.
5. Allow GPT as Judge to analyze this and return an explainable validation to user.

## Project Structure

```
BMI-Estimator-with-Reasoning/
|-- app.py                          # CLI entry point: python app.py photo.jpg
|-- test_dataset.py                 # Dataset access validation
|-- bmi_system.py                   # System-level orchestration (WIP)
|-- requirements.txt                # Python dependencies
|-- config/
|   |-- settings.py                 # Centralized configuration (paths, model names, seed)
|-- src/
|   |-- models/
|   |   |-- pose_estimator.py       # MediaPipe pose detection + ratio extraction
|   |   |-- bmi_predictor.py        # Random Forest inference + WHO categorization
|   |-- data/
|       |-- load_real_data.py       # BEDLAM + UniqueData pipeline, synthetic fallback
|-- scripts/
|   |-- train_with_real_data.py     # Model training + evaluation script
|   |-- load_datasets.py            # Dataset exploration utilities
|   |-- explore_bedlam.py           # BEDLAM data structure exploration
|-- models/
|   |-- pretrained/
|       |-- checkpoints/            # Serialized model + scaler (.pkl)
|-- data/
    |-- bmi_dataset.csv             # Synthetic training data
    |-- combined_dataset.csv        # Combined BEDLAM + synthetic
```

## Installation




## Dependencies

| Package | Purpose |
|---------|---------|
| PyTorch 2.0+ / torchvision | Foundation for future deep learning models |
| MediaPipe 0.10+ | Pose landmark detection (33-point model) |
| OpenCV 4.8+ | Image loading and color space conversion |
| scikit-learn 1.3+ | Random Forest model, preprocessing, evaluation |
| pandas 2.0+ | Dataset manipulation and CSV I/O |
| HuggingFace datasets / hub | Remote dataset access pipeline |


## Design Decisions



## License

This project is a work in progress. License to be determined.

---

*Status: Active Development -- features and performance metrics are evolving.*
