import os
import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
import torch.nn as nn

def preprocess_image_cv2(image_path): # apply CLAHE and green channel extraction
    img = cv2.imread(image_path)  # read the image
    green_channel = img[:, :, 1]  # extract the green channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # create CLAHE object
    enhanced_img = clahe.apply(green_channel)  # apply CLAHE to the green channel
    return enhanced_img

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetClassifier, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # load a ResNet-18 model
        num_features = self.model.fc.in_features  # get the number of features in the last layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),  # dropout with 50% probability
            nn.Linear(num_features, 1)  # fully connected layer for classification
        )

    def forward(self, x):
        return self.model(x)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ensure images are 224x224 pixels
    transforms.ToTensor(),  # convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # imageNet normalization
])

def load_model():
    model = ResNetClassifier(num_classes=2)
    checkpoint = torch.load('best_model.pth', weights_only=True)  # load the best model
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def predict_image(image, model):
    image = Image.fromarray(image).convert('RGB')  # ensure RGB format
    image = transform(image).unsqueeze(0)  # apply training transformations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)  # move to the appropriate device

    with torch.no_grad():
        output = model(image)
        probability = torch.sigmoid(output).item()  # apply sigmoid for binary classification

    threshold = 0.5
    return 1 if probability > threshold else 0

def main():
    st.set_page_config(page_title="Diabetic Retinopathy Classifier", page_icon="👀", layout="wide")
    url = 'https://github.com/bsgelman/Diabetic-Retinopathy-Classification'
    with st.sidebar:
        st.header("ABOUT")
        st.markdown("""
        This app uses a Convolutional Neural Network (CNN) to classify retinal images as either **No Diabetic Retinopathy** 
        or **Diabetic Retinopathy Detected**. This is done using a ResNet-18 model.
        
        **Steps to Use:**
        1. Upload an image of the retina.
        2. View the prediction.
        3. Learn more about the model [here](%s).
        """ % url)

    st.title("Diabetic Retinopathy Classification")

    uploaded_file = st.file_uploader("Upload a Retina Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image_path = "temp_image.png"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        processed_image = preprocess_image_cv2(image_path)

        model = load_model()

        prediction = predict_image(processed_image, model)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.header("Uploaded Image")
            st.image(uploaded_file, caption="Uploaded Retina Image", width=300, output_format="PNG")
        with col2:
            st.header("Prediction")
            if prediction == 1:
                st.error("##### Diabetic Retinopathy Detected")
            else:
                st.success("##### No Diabetic Retinopathy")
        with col3:
            st.write(":red[WARNING: This is not a diagnosis and make sure to consult a healthcare professional.]")
    else:
        st.warning("Please upload a retina image to proceed.")

if __name__ == "__main__":
    main()
