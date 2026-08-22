# BMI Estimator with Reasoning

This project validates the model's BMI predictions with GPT as Judge and Conformal Prediction Regression. Providing the user with an explicable validation pipeline.

## Research Foundation

This project translates published peer-reviewed research into a functional prototype. The methodology is grounded in three key works:

- Sui, J. C., Zhao, et al. — *Body Weight Estimation Using Virtual Anthropometric Measurements from a Single Image*
- Manichand, F. R., et al. — *Digital Scale: Open-Source On-Device BMI Estimation from Smartphone Camera Images Trained on a Large-Scale Real-World Dataset*
- Jin, Z., Huang, J., Wang, W., et al. — *Estimating Human Weight from a Single Image*

Building on the Digital Scale approach, this project extends the pipeline with conformal prediction regression validation and **GPT-as-Judge reasoning layer** that provides natural-language confidence scoring and clinical context alongside each statistical guarantee. Making the output interpretable by technical/non-technical users.

## Problem Statement



## How It Works


## Technical Architecture


## Model Performance



## Project Structure

```
BMI-Estimator-with-Reasoning/
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



## Design Decisions



## License

This project is a work in progress. License to be determined.

---

*Status: Active Development — features and performance metrics are evolving.*
