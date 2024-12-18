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

# Function to preprocess images with CLAHE and green channel extraction
def preprocess_image_cv2(image_path):
    """Apply CLAHE and Green Channel Extraction."""
    img = cv2.imread(image_path)  # Read the image
    green_channel = img[:, :, 1]  # Extract the green channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Create CLAHE object
    enhanced_img = clahe.apply(green_channel)  # Apply CLAHE to the green channel
    return enhanced_img

# Load the ResNet model
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetClassifier, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # Load a ResNet-18 model
        num_features = self.model.fc.in_features  # Get the number of features in the last layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),  # Dropout with 50% probability
            nn.Linear(num_features, 1)  # Fully connected layer for classification
        )

    def forward(self, x):
        return self.model(x)

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure images are 224x224 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load the trained model
def load_model():
    model = ResNetClassifier(num_classes=2)
    checkpoint = torch.load('best_model.pth', weights_only=True)  # Load the best model
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def predict_image(image, model):
    # Convert to RGB and apply transforms
    image = Image.fromarray(image).convert('RGB')  # Ensure RGB format
    image = transform(image).unsqueeze(0)  # Apply training transformations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)  # Move to the appropriate device

    # Forward pass through the model
    with torch.no_grad():
        output = model(image)
        probability = torch.sigmoid(output).item()  # Apply sigmoid for binary classification

    # Apply threshold to classify
    threshold = 0.5
    return 1 if probability > threshold else 0


# Streamlit web app
def main():
    st.set_page_config(page_title="Diabetic Retinopathy Classifier", page_icon="👀", layout="wide")
    url = 'https://github.com/bsgelman/Diabetic-Retinopathy-Classification'
    # Sidebar information
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

    # File upload section
    uploaded_file = st.file_uploader("Upload a Retina Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Save the uploaded image temporarily
        image_path = "temp_image.png"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process the image
        processed_image = preprocess_image_cv2(image_path)

        # Load the model
        model = load_model()

        # Make prediction
        prediction = predict_image(processed_image, model)

        # Display results in columns
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
