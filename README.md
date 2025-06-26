# Cone Detection with TensorFlow Lite + Coral Edge TPU

Train and deploy a cone detector using TensorFlow Lite Model Maker and compile it for the Coral Edge TPU.

## 📁 Structure

.
├── images/ # Training images + Pascal VOC XMLs
├── train.py # Train and export model
├── verify_if_int8.py # Check if model is quantized
├── exported-model-vX/ # TFLite output dir
├── model_edgetpu.tflite # Compiled TPU model

## 🧪 Quickstart

1. **Setup** (Python 3.9)
```bash
python3 -m venv tflite-venv
source tflite-venv/bin/activate
pip install tflite-model-maker==0.4.3 tensorflow==2.8 pycocotools
Train

bash
Copy
Edit
python3 train.py
Check Quantization

bash
Copy
Edit
python3 verify_if_int8.py
Compile for Coral

bash
Copy
Edit
edgetpu_compiler exported-model-vX/model.tflite
