# Digital Scale - Quick Start Guide

This folder contains a streamlined version of the Digital Scale BMI estimation system for quick setup and testing. It focuses on running forward passes (inference) with pre-trained models.

## 📁 Contents

- `model.py` - DenseNet model definitions (SEDenseNet121 and SEDenseNet201)
- `dataset.py` - Dataset handling and preprocessing utilities
- `predict_bmi.py` - Main inference script for BMI prediction
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules for weights and data

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Required Files

**⚠️ Important:** This folder does not include model weights or sample data. You need to download them separately:

**Option A: Download from Authors (Recommended)**
Contact the authors to obtain the complete dataset and model weights:
- **Email**: rmanichand@ethz.ch or planger@ethz.ch
- **Request**: "Digital Scale model weights and sample data"

**Option B: Use Your Own Data**
If you have your own images, structure them as follows:
```
data/
├── visual-body-to-bmi.csv  # CSV with columns: individual_id, bmi, height_in, weight_lb, is_female, image_path
└── individual_folders/     # Folders named by individual_id
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### 3. Organize Files

Place the downloaded files in the following structure:
```
get_started/
├── weights/
│   └── best_model.ckpt     # Model checkpoint file
├── data/
│   ├── visual-body-to-bmi.csv
│   ├── individual_folder_1/
│   ├── individual_folder_2/
│   └── ...
├── model.py
├── dataset.py
├── predict_bmi.py
└── requirements.txt
```

## 🔍 Usage

### Basic Usage

Run BMI prediction on sample data:

```bash
python predict_bmi.py
```

### Advanced Usage

Customize the prediction with various options:

```bash
# Use DenseNet201 instead of DenseNet121
python predict_bmi.py --model_type densenet201

# Specify custom paths
python predict_bmi.py --model_path weights/my_model.ckpt --data_dir my_data/

# Force CPU usage
python predict_bmi.py --device cpu

# Use larger batch size (if you have enough GPU memory)
python predict_bmi.py --batch_size 8
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model_path` | `weights/best_model.ckpt` | Path to model checkpoint |
| `--model_type` | `densenet121` | Model architecture (`densenet121` or `densenet201`) |
| `--data_dir` | `data` | Directory containing dataset |
| `--batch_size` | `1` | Batch size for inference |
| `--device` | `auto` | Computation device (`auto`, `cuda`, or `cpu`) |

## 📊 Expected Output

When running successfully, you should see output like:

```
Using device: cuda
Loading dataset...
Found 10 valid images for testing
Loading model...
Loaded densenet121 model from weights/best_model.ckpt

Running BMI predictions...
--------------------------------------------------------------------------------
ID: 1afea7     | Predicted: 38.06 | Actual: 38.51 | Error: 0.45
ID: 1afea7     | Predicted: 27.22 | Actual: 28.50 | Error: 1.28
ID: 1adchl     | Predicted: 32.70 | Actual: 32.32 | Error: 0.38
...
--------------------------------------------------------------------------------
Summary Statistics (n=10):
Average Error: 0.49
Max Error: 1.28
Min Error: 0.14

Completed predictions for 10 images.
```

## 🛠️ Troubleshooting

### Common Issues

**1. "Dataset CSV not found"**
```
Dataset CSV not found at data/visual-body-to-bmi.csv
Please download the sample data from the authors:
Contact: rmanichand@ethz.ch or planger@ethz.ch
```
**Solution**: Download the dataset from the authors or check your data directory structure.

**2. "Model checkpoint not found"**
```
Model checkpoint not found at weights/best_model.ckpt
Please download the model weights from the authors:
Contact: rmanichand@ethz.ch or planger@ethz.ch
```
**Solution**: Download the model weights from the authors or check your weights directory.

**3. "No valid images found"**
```
No valid images found. Please check your data directory structure.
Expected structure:
  data/
    visual-body-to-bmi.csv
    individual_id_folders/
      *.jpg or *.png files
```
**Solution**: Ensure your data follows the expected directory structure.

**4. CUDA Out of Memory**
- Use `--device cpu` to run on CPU
- Reduce `--batch_size` to 1 (default)
- Close other GPU-intensive applications

## 🔬 Model Details

### SEDenseNet121
- **Architecture**: DenseNet121 with Squeeze-and-Excitation blocks
- **Input Size**: 224x224 RGB images
- **Output**: Single BMI value (regression)
- **Parameters**: ~7M parameters
- **Performance**: MAPE ~7.9% on WayBED dataset

### SEDenseNet201
- **Architecture**: DenseNet201 with Squeeze-and-Excitation blocks  
- **Input Size**: 224x224 RGB images
- **Output**: Single BMI value (regression)
- **Parameters**: ~18M parameters
- **Performance**: Potentially better accuracy but slower inference

## 🌟 Next Steps

After getting familiar with this quick start:

1. **Explore the full repository** for training scripts, evaluation tools, and mobile deployment
2. **Check the main README** for comprehensive documentation
3. **Try the CLAID mobile app** for real-time BMI estimation
4. **Experiment with your own datasets** following the data structure guidelines

## 📄 License

This project is part of the Digital Scale research. Please refer to the main repository for license information.

## 📧 Support

For questions, issues, or to request model weights and sample data:
- **Primary Contact**: rmanichand@ethz.ch
- **Secondary Contact**: planger@ethz.ch

---

**Note**: This quick start guide is designed to get you running BMI prediction quickly. For training new models, comprehensive evaluation, or mobile deployment, please refer to the main repository documentation. 