# BMI Estimator with Reasoning

Uncertainty-aware BMI estimation from a single image -- Conformal prediction + GPT-as-Judge reasoning

Most BMI-from-image models output a single number with no indication of how trustworthy it is. This project adds two layers on top of a DenseNet regressor: a conformal prediction layer that turns the point estimate into a statistically calibrated confidence interval, and a GPT-as-Judge layer that explains the result in plain English.
    Instead of "BMI = 27.3", the system says "BMI = 27.3, 90% interval [20.9, 33.7], and here's how much you should trust it."

## Research Foundation

This project translates published peer-reviewed research into a functional prototype. The methodology is grounded in three key works:

- Sui et al. — Body Weight Estimation Using Virtual Anthropometric Measurements from a Single Image
- Manichand et al. — Digital Scale: Open-Source On-Device BMI Estimation from Smartphone Camera Images Trained on a Large-Scale Real-World Dataset
- Jin et al. — Estimating Human Weight from a Single Image

Building on the Digital Scale approach (SE-DenseNet image-to-BMI regression), this project extends the pipeline with split conformal prediction for distribution-free uncertainty quanitification and a GPT-as-Judge reasoning layer for interpretability.


## How It Works

1. Regression - a Squeeze-and-Excitation DenseNet (SE-DenseNet121) in pyTorch maps a single 224x224 image to a BMI point estimate.
2. Conformal calibration - on a held-out calibration set, the absolute residuals are collected and q is taken as their 90th-percentile. Every prediction becomes an interval.
3. Reasoning - GPT-4.1-Nano receives the estimate, interval, and demographic context, and returns a plain-English explanation with an honest confidence read.

## Model Performance

Evaluated on Celeb-FBI (7,208 real images; 6,174 with valid weight/height labels), on a held-out test set of 620 samples:

| Metric              | Value              |
| MAE                 | 2.81 BMI Points    |
| MAPE                | 11.56%             |
| Conformal q         | 6.44               |
| Verified Coverage   | 90.4 % (target 90%)|

Best case: id 1589 -- predicted 19.93, actual 19.93. Worst case: id 7117 -- predicted 33.48, actual 61.23 (an extreme BMI the model under-predicts; see Limitations)

## Example Reasoning

For a typical sample (predicted 19.1, 90% interval [12.7, 25.5]):
    "The estimate suggests a BMI around 19.1, which is in the health range, but the true BMI could be anywhere from about 12.7 to 25.5. Since the range is quite wide, there's a lot of uncertainty, so this number should be taken as a rough guess. Keep in mind that this is just an estimate and not a precise measurement."

## Project Structure

```
BMI-Estimator-with-Reasoning/
|-- get_started/                    # Active code (BMI Estimator w/Reasoning)
|   |-- dataset.py                  # Celeb-FBI loading + preprocessing (BMIDataset)
|   |-- model.py                    # SE-DenseNet architecture
|   |-- predict_bmi.py              # Inference (point estimate + conformal interval)
|   |-- train_bmi.py                # Training loop
|   |-- calibrate_bmi.py            # Conformal calibration -> q_hat
|   |-- explain_bmi.py              # GPT-as-Judge reasoning layer
|   |-- requirements.txt            
|-- src/                            # Upstream DigitalScale reference code
|-- weights/                        # best_model.ckpt + q_hat.npy
```

## Installation

bash
git clone https://github.com/T-Thomas1/BMI-Estimator-with-Reasoning
cd BMI-Estimator-with-Reasoning
pip install -r get_started/requirements.txt

## Usage

bash
python get_started/calibrate_bmi.py
python get_started/predict_bmi.py
python get_started/explain_bmi.py

## Limitations

- Constant interval width - split conformal gives every sample the same q. This motivates CQR as the next step.
- Under-predicts extreme BMIs.
- Not a medical device.

## Roadmap

- Conformalized Quantile Regression (CQR) for adaptive, per-image interval widths.
- A photo-uoload interface wrapping the reasoning layer for interactive use.

## License

Work in progress -- license to be determined.

---

*Status: Active Development*
