# BMI Estimator with Reasoning

A ML system that takes DigitalScale's DenseNet architecture and implements it on a synthetic BEDLAM dataset. This project validates the model's predictions for BMI with a natural language reasoning.

## Research Foundation

This project translates published peer-reviewed research into a functional prototype. The pose-to-anthropometric-ratio methodology is grounded in three key works:

- Sui, J. C., Zhao, et al. — *Body Weight Estimation Using Virtual Anthropometric Measurements from a Single Image*
- Manichand, F. R., et al. — *Digital Scale: Open-Source On-Device BMI Estimation from Smartphone Camera Images Trained on a Large-Scale Real-World Dataset*
- Jin, Z., Huang, J., Wang, W., et al. — *Estimating Human Weight from a Single Image*

Building on the Digital Scale approach (MediaPipe pose → body ratios → ML regression), this project extends the pipeline with a novel **GPT-as-Judge reasoning layer** that provides natural-language confidence scoring and clinical context alongside each numeric BMI estimate — making the output interpretable by non-technical users.

## Problem Statement

<<<<<<< HEAD
Recently tools available during telehealth, or fitness tracking have been readily available. DigitalScale provides photo BMI estimation. This project investigates the final model performance through training, calibration, and evaluation. Conformal Prediction allows a statistical guarantee of surity that the model's prediction is correct. Paired with GPT-as-Judge provides an explainable validation pipeline.
=======
Traditional BMI calculation requires a scale and measuring tape — tools not always available in telehealth, fitness tracking, or resource-constrained settings. This project investigates whether a single 2D image contains enough anthropometric signal to produce a meaningful BMI estimate, and whether AI reasoning can add clinical context to a numeric output.
>>>>>>> f288db5b622b9d199cd4de0d877424947353ccee

## How It Works

The system operates in three stages:

<<<<<<< HEAD
1. Train The model
2. Run The model on calibrated data
3. Calculate absolute non-conformity
4. Find Q_Hat threshold and apply to our new data.
5. Allow GPT as Judge to analyze this and return an explainable validation to user.
=======
1. **Pose Estimation** — MediaPipe extracts 33 body landmarks from the input image, localizing shoulders, hips, nose, and ankles.
2. **Ratio Extraction** — Two body ratios are computed from the landmark coordinates: Shoulder-to-Waist Ratio (SWR) and Waist-to-Height Ratio (WHR). These ratios serve as the feature vector, normalizing for camera distance and image resolution.
3. **BMI Prediction** — A scikit-learn Random Forest regressor maps the (SWR, WHR) pair to a BMI estimate, which is then categorized into standard WHO weight classes (Underweight, Normal Weight, Overweight, Obese).

## Technical Architecture

```
Input Image (RGB)
       |
       v
[Pose Estimation] --> MediaPipe Pose (33 landmarks)
       |
       v
[Ratio Extraction] --> SWR (shoulder_width / hip_width)
                   --> WHR (hip_width / height)
       |
       v
[BMI Predictor] --> Random Forest Regressor + StandardScaler
       |
       v
Output: BMI value + WHO category
```

## Model Performance

The Random Forest model is trained on BEDLAM 3D body measurement data, with planned integration of UniqueData/body-measurements-dataset for improved real-world generalization.

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest Regressor (n=100, max_depth=10) |
| Features | SWR, WHR |
| Test Split | 20% holdout (random_state=42) |
| Performance | Reported at training time (MAE, R-squared) |

Performance varies with image quality, pose angle, clothing, and occlusions. Front-facing, full-body photos with minimal clothing produce the most reliable results.
>>>>>>> f288db5b622b9d199cd4de0d877424947353ccee

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

*Status: Active Development — features and performance metrics are evolving.*
