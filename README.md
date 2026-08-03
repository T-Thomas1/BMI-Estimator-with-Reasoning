# BMI Estimator with Reasoning

An AI system that estimates Body Mass Index (BMI) from a single full-body photograph using pose estimation, anthropometric ratio extraction, and machine learning inference. This project explores the intersection of computer vision and health metrics without requiring specialized hardware.

## Problem Statement

Traditional BMI calculation requires a scale and measuring tape -- tools not always available in telehealth, fitness tracking, or resource-constrained settings. This project investigates whether a single 2D image contains enough anthropometric signal to produce a meaningful BMI estimate, and whether AI reasoning can add clinical context to a numeric output.

## How It Works

The system operates in three stages:

1. **Pose Estimation** -- MediaPipe extracts 33 body landmarks from the input image, localizing shoulders, hips, nose, and ankles.
2. **Ratio Extraction** -- Two body ratios are computed from the landmark coordinates: Shoulder-to-Waist Ratio (SWR) and Waist-to-Height Ratio (WHR). These ratios serve as the feature vector, normalizing for camera distance and image resolution.
3. **BMI Prediction** -- A scikit-learn Random Forest regressor maps the (SWR, WHR) pair to a BMI estimate, which is then categorized into standard WHO weight classes (Underweight, Normal Weight, Overweight, Obese).

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

```bash
git clone https://github.com/T-Thomas1/BMI-Estimator-with-Reasoning.git
cd BMI-Estimator-with-Reasoning
pip install -r requirements.txt
```

## Usage

### Quick Start (with pre-trained model)

```bash
python app.py path/to/photo.jpg
```

Output:
```
BMI: 24.3 (Normal weight)
Shoulder/Hip: 1.351
Waist/Height: 0.472
```

### Training from Scratch

```bash
python scripts/train_with_real_data.py
```

The training script loads BEDLAM 3D measurements, combines them with any available UniqueData samples, trains the Random Forest model, reports MAE and R-squared, and saves the model to `models/pretrained/checkpoints/bmi_model.pkl`.

## Dependencies

| Package | Purpose |
|---------|---------|
| PyTorch 2.0+ / torchvision | Foundation for future deep learning models |
| MediaPipe 0.10+ | Pose landmark detection (33-point model) |
| OpenCV 4.8+ | Image loading and color space conversion |
| scikit-learn 1.3+ | Random Forest model, preprocessing, evaluation |
| pandas 2.0+ | Dataset manipulation and CSV I/O |
| HuggingFace datasets / hub | Remote dataset access pipeline |

## Roadmap

This is an active work in progress. Planned improvements:

- [ ] Integrate UniqueData/body-measurements-dataset for real-world training data
- [ ] Add LLaVA (llava-1.5-7b) for multimodal reasoning on BMI predictions
- [ ] Integrate SAM (Segment Anything Model) for foreground person segmentation
- [ ] Build a Gradio or Streamlit web interface for interactive demos
- [ ] Add confidence scoring alongside numeric predictions
- [ ] Improve robustness to varied poses, clothing, and lighting conditions

## Design Decisions

**Why body ratios instead of pixel-to-BMI regression?** Raw pixel approaches overfit to camera parameters (distance, focal length, resolution). Body ratios normalize these factors, allowing the model to generalize across different photo conditions.

**Why Random Forest?** For a two-feature problem on moderate data, ensemble tree methods provide interpretable results with strong baseline performance. A neural approach is planned once dataset size justifies the added complexity.

**Why MediaPipe?** MediaPipe provides production-grade pose estimation that runs on CPU, making the pipeline accessible without GPU hardware for inference.

## License

This project is a work in progress. License to be determined.

---

*Status: Active Development -- features and performance metrics are evolving.*
