# BMI Estimator with Reasoning

This project validates the model's BMI predictions with GPT as Judge and Conformal Prediction Regression. Providing the user with an explicable validation pipeline.

## Research Foundation

This project translates published peer-reviewed research into a functional prototype. The methodology is grounded in three key works:

- Sui, J. C., Zhao, et al. — *Body Weight Estimation Using Virtual Anthropometric Measurements from a Single Image*
- Manichand, F. R., et al. — *Digital Scale: Open-Source On-Device BMI Estimation from Smartphone Camera Images Trained on a Large-Scale Real-World Dataset*
- Jin, Z., Huang, J., Wang, W., et al. — *Estimating Human Weight from a Single Image*

Building on the Digital Scale approach, this project extends the pipeline with conformal prediction regression validation and **GPT-as-Judge reasoning layer** that provides natural-language confidence scoring and clinical context alongside each statistical guarantee. Making the output interpretable by technical/non-technical users.

## Problem Statement

A DenseNet201 model provides strong predictions with images. This project takes a heuristical approach to DenseNet's model. Testing the coverage of its predictions by utilizing a coverage guarantee. How will uncertainty quantifiction wrap DenseNet's model in a calibrated layer? 

## How It Works


## Technical Architecture


## Model Performance

Summary Statistics (n=620):

Average Error (MAE): 2.81

MAPE: 11.56%

Max Error: 27.75
Min Error: 0.00

BEST (id 1589): pred 19.93 | actual 19.93
WORST (id 7117): pred 33.48 | actual 61.23 #This is an actual entry for an obese person within the Database


## Project Structure

```
BMI-Estimator-with-Reasoning/
|-- data                            # Not Needed (DigitalScale)
|-- docs/                           # Not Needed (DigitalScale)
|-- get_started/                    # Quick start guide with minimal setup for BMI Prediction
    |-- dataset.py
    |-- model.py                    # DenseNet architecture (DigitalScale)
    |-- predict_bmi.py              # BMI Prediction on images (BMI Estimator w/Reasoning)
    |-- train_bmi.py                # Training loop (BMI Estimator w/Reasoning)
|-- models/
|   |-- pretrained/
|       |-- checkpoints/            # Serialized model + scaler (.pkl) (BMI Estimator w/Reasoning)
|-- notebooks/                      # Posutre clustering example (DigitalScale)
|-- src/                            # Source Code (DigitalScale)
|-- weights/                        # Pre-trained checkpoints for model (BMI Estimator w/Reasoning)
```

## Installation




## Dependencies

| Package | Purpose |
|---------|---------|

Tensorflow[and-cuda] | Use of GPU during training and prediction

## Design Decisions

Tensorflow[and-cuda]

## License

This project is a work in progress. License to be determined.

---

*Status: Active Development — features and performance metrics are evolving.*
