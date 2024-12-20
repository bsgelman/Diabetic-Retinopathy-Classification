# Diabetic Retinopathy Detection

This repository contains three key components designed to classify retinal images for diabetic retinopathy detection. The project supports both binary classification (presence or absence of diabetic retinopathy) and multi-class classification (severity levels). A Streamlit web application provides an interface for users to upload images and get real-time predictions.

---

## Components

### **1. app.py**
A Streamlit-based web application for running the binary classification model.

- **Features**:
  - **Image Upload**: Users can upload retinal images (PNG, JPG, JPEG) for analysis.
  - **Real-Time Prediction**: Uses a pre-trained ResNet-18 model to classify images as:
    - **No Diabetic Retinopathy**
    - **Diabetic Retinopathy Detected**
  - **Preprocessing**:
    - Extracts the green channel of the image.
    - Enhances the image using CLAHE (Contrast Limited Adaptive Histogram Equalization).
  - **User-Friendly Interface**:
    - Sidebar with instructions and an external link for further information.
    - Visual display of uploaded images and prediction results.

- **Usage**:
  1. Run the Streamlit app:
     ```bash
     streamlit run app.py
     ```
  2. Upload a retinal image and view the prediction.

---

### **2. BinaryDetection.py**
A script for training a binary classification model using the ResNet-18 architecture.

- **Features**:
  - **Dataset Handling**:
    - Automatically splits the dataset into training (64%), validation (16%), and testing (20%) sets.
    - Processes images using CLAHE and green channel extraction.
  - **Model Architecture**:
    - ResNet-18 with modifications for binary classification.
    - Includes dropout and fully connected layers for enhanced performance.
  - **Training**:
    - Implements early stopping to prevent overfitting.
    - Tracks and displays training and validation accuracy/loss per epoch.
  - **Testing**:
    - Evaluates the model on the test dataset.
    - Displays final test accuracy and loss.

- **Usage**:
  1. Ensure the dataset is properly organized.
  2. Run the script to train the model:
     ```bash
     python BinaryDetection.py
     ```

---

### **3. DRDetection.py**
A script for training a multi-class classification model to categorize images into severity levels:
  - **No_DR**
  - **Mild**
  - **Moderate**
  - **Severe**
  - **Proliferate_DR**

- **Features**:
  - **Dataset Handling**:
    - Similar to `BinaryDetection.py`, but tailored for multi-class classification.
    - Splits the dataset into training, validation, and testing sets.
  - **Model Architecture**:
    - ResNet-18 modified for five-class classification.
    - Uses CrossEntropyLoss for multi-class learning.
  - **Training**:
    - Includes early stopping.
    - Tracks training and validation metrics per epoch.
  - **Testing**:
    - Evaluates the model on the test dataset and reports accuracy and loss.

- **Usage**:
  1. Ensure the dataset is properly organized.
  2. Run the script to train the model:
     ```bash
     python DRDetection.py
     ```

---

## Dataset

This project uses the [Diabetic Retinopathy 224x224 Gaussian Filtered Dataset](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered?select=gaussian_filtered_images) from Kaggle. The dataset contains retinal images that have been preprocessed using a Gaussian filter and resized to 224x224 pixels, making it suitable for deep learning models.

---

## Prerequisites

- **Hardware**:
  - A system with GPU is recommended for training.
- **Libraries**:
  - Python 3.8+
  - PyTorch, torchvision, Streamlit, OpenCV, and NumPy

---

## Notes

- The Streamlit application is designed for binary classification; modify it if you wish to use the multi-class model.
