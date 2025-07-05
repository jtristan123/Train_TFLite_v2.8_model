# CCC Cone Detection with Coral Edge TPU

This project demonstrates how to train, convert, and deploy a custom object detection model (cone detector) on the Google Coral USB Accelerator using TensorFlow Lite and Edge TPU.

#### 📦 What This Project Does

- Trains a MobileNet SSD model to detect cones using custom images.
- Converts and compiles the model for Coral Edge TPU.
- Runs live object detection using Raspberry Pi and a USB webcam with Picamera2.
- Sends control commands to a robot via serial based on object detection.

---

#### 🛠️ Environment Setup

### 🖥️ Training (on local PC using WSL + VSCode)
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/ccc-coral-cone-detection
   cd ccc-coral-cone-detection
2. Create a Python 3.9 virtual environment (Coral TPU is not compatible with 3.11):

```bash
sudo apt install python3.9 python3.9-venv python3.9-dev
python3.9 -m venv coral-env
source coral-env/bin/activate
```
3. Install training dependencies:

```bash
pip install -r requirements.txt
```
4. Train your model:

```bash
python3 train.py
```
Training uses tflite_model_maker and image-label pairs in images/ folder.<br>
####📂 Folder Structure
```
├── images/                   # Training images (jpg) and labels (xml)
├── exported-model-v/         # Exported and compiled TFLite model goes here from train.py, this can all be moved to raspberry 
   ├── model.tflite           # .tflite model from train.py before compiler
   ├── labels.txt             # dont forget the labels.txt file  
├── train.py                  # Training and exports the model .tflite script
├── verify_if_int8.py         # verify the model is in int8 compabily with coral TFlite 


```

####📦 Model Conversion
After training, verify your model is TensorFlow Lite:

```bash
python3 verify_if_int8.py
```
Then compile it for Edge TPU (on a Linux PC or in WSL):

```bash
edgetpu_compiler model.tflite
```
Output: model_edgetpu.tflite
move this file to the same folder as your model.tflite

