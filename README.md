# 😷 Face Mask Detection using CNN

A Deep Learning based Face Mask Detection system that classifies whether a person is wearing a mask or not.

The project uses a custom Convolutional Neural Network (CNN) built with PyTorch and provides an interactive web application using Streamlit.

## 🚀 Live Demo

🔗 Streamlit App:

https://facemaskdetection23.streamlit.app/

---

# 📌 Project Overview

Face Mask Detection is a computer vision application that uses Deep Learning to identify mask usage from facial images.

The model takes an input face image, processes it through a CNN architecture, and predicts one of two classes:

- 😷 With Mask
- 😐 Without Mask

---

# ✨ Features

✅ Upload face images  
✅ CNN based image classification  
✅ PyTorch implementation  
✅ Real-time prediction using Streamlit  
✅ Confidence score display  
✅ Lightweight deployment  
✅ User-friendly web interface  

---

# 🧠 Deep Learning Workflow

```
Input Image
      |
      ↓
Image Preprocessing
(Resize 224x224 + Normalization)
      |
      ↓
CNN Feature Extraction
(Convolution + ReLU + Pooling)
      |
      ↓
Fully Connected Layers
      |
      ↓
Softmax Classification
      |
      ↓
Prediction
(With Mask / Without Mask)
```

---

# 🏗️ Model Architecture

Custom CNN Architecture:

```
Input Image
    |
Conv2D (32 filters)
    |
ReLU Activation
    |
Max Pooling

    |
Conv2D (64 filters)
    |
ReLU Activation
    |
Max Pooling

    |
Conv2D (128 filters)
    |
ReLU Activation
    |
Max Pooling

    |
Flatten

    |
Fully Connected Layer
512 neurons

    |
Dropout

    |
Output Layer
2 Classes
```

---

# 📂 Project Structure

```
Face-Mask-Detection-using-CNN

│
├── app.py
│
├── requirements.txt
│
├── models
│    └── best_face_mask_model.pth
│
├── notebooks
│    └── Face Mask Detection.ipynb
│
├── dataset
│
└── README.md
```

---

# 📊 Dataset

Dataset contains two classes:

```
with_mask
without_mask
```

Classes:

| Class | Description |
|---|---|
| 😷 with_mask | Person wearing mask |
| 😐 without_mask | Person not wearing mask |

---

# ⚙️ Technologies Used

### Programming Language
- Python

### Deep Learning
- PyTorch
- Torchvision

### Computer Vision
- PIL
- Image Processing

### Deployment
- Streamlit

### Development Environment
- Google Colab
- Google Drive

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/dipanshu14324/-Face-Mask-Detection-using-CNN.git
```

Go inside project folder:

```bash
cd Face-Mask-Detection-using-CNN
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Application Locally

Run Streamlit:

```bash
streamlit run app.py
```

Open browser:

```
http://localhost:8501
```

---

# 🏋️ Training Details

Model trained using:

- Framework: PyTorch
- Input Size: 224 × 224
- Optimizer: Adam
- Loss Function: Cross Entropy Loss
- Epochs: 10

Training pipeline:

```
Dataset
   |
Data Loading
   |
Image Transformation
   |
CNN Training
   |
Validation
   |
Model Saving
```

---

# 📈 Model Performance

Training Results:

```
Training Accuracy : ~99%
Validation Accuracy : ~95%
```

The trained model is saved as:

```
best_face_mask_model.pth
```

---

# 🌐 Deployment

The application is deployed using:

```
GitHub
   |
   ↓
Streamlit Cloud
   |
   ↓
Live Web Application
```

Streamlit deployment allows users to upload images and get predictions directly from the browser. :contentReference[oaicite:1]{index=1}

---

# 🔮 Future Improvements

- Real-time webcam detection
- Face detection using OpenCV
- YOLO based mask detection
- Mobile application deployment
- Better dataset augmentation

---

# 👨‍💻 Author

**Dipanshu Shukla**

Deep Learning | Computer Vision | PyTorch

GitHub:
https://github.com/dipanshu14324

# 🚀 Live Demo

Try the deployed application here:

🔗 **Streamlit App:**  
https://facemaskdetection23.streamlit.app/

Upload a face image and the model will predict:

- 😷 With Mask
- 😐 Without Mask

with confidence score.