import os
import random
import time
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from shutil import copy2
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

def preprocess_image_cv2(image_path):
    #Apply CLAHE and Green Channel Extraction
    img = cv2.imread(image_path)  # read the image
    green_channel = img[:, :, 1]  # extract the green channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # create CLAHE object
    enhanced_img = clahe.apply(green_channel)  # apply CLAHE to the green channel
    return enhanced_img

class CustomImageDataset(Dataset):
    def __init__(self, image_folder, transform=None, preprocess=None):
        self.image_folder = ImageFolder(image_folder)  # use ImageFolder for labels
        self.transform = transform
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        image_path, label = self.image_folder.samples[idx]  # get image path and label
        if self.preprocess:
            # preprocess with custom function
            image = preprocess_image_cv2(image_path)
            image = Image.fromarray(image).convert('L')  # convert back to PIL image in grayscale
            image = image.convert('RGB')  # convert grayscale back to RGB for compatibility
        else:
            # load original image
            image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def split_dataset(data_dir, seed=10):
    random.seed(seed)

    train_ratio = 0.64  # 64% for training
    val_ratio = 0.16  # 16% for validation
    #test_ratio = 0.20  # 20% for testing

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    if os.path.exists(train_dir) or os.path.exists(val_dir) or os.path.exists(test_dir):
        print("WARNING: Training, validation, and/or testing directories already exist.")
        return
    os.makedirs(train_dir)
    os.makedirs(test_dir)
    os.makedirs(val_dir)

    for class_name in os.listdir(data_dir):  # iterate through each severity folder
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir) and class_name not in ['train', 'val', 'test', 'Binary']:
            images = os.listdir(class_dir)
            random.shuffle(images)
            #images = images[:708]

            train_size = int(len(images) * train_ratio)
            val_size = int(len(images) * val_ratio)
            train_images = images[:train_size]
            val_images = images[train_size:train_size + val_size]
            test_images = images[train_size + val_size:]  # remainder of images for testing

            # create subdirectories for each class in train, val, and test folders
            train_class_dir = os.path.join(train_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)

            os.makedirs(train_class_dir)
            os.makedirs(val_class_dir)
            os.makedirs(test_class_dir)

            # copy files to respective directories
            for img in train_images:
                copy2(os.path.join(class_dir, img), os.path.join(train_class_dir, img))
            for img in val_images:
                copy2(os.path.join(class_dir, img), os.path.join(val_class_dir, img))
            for img in test_images:
                copy2(os.path.join(class_dir, img), os.path.join(test_class_dir, img))

    print("\033[1;32mData split completed.\033[0m")

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, save_path='best_model.pth'):
        # patience: how long to wait after last time validation loss improved
        # verbose: if True, prints a message for each validation loss improvement
        # delta: Minimum change in the monitored quantity to qualify as an improvement
        # save_path: Path to save the best model
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.save_path = save_path
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Validation loss decreased to {val_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNetClassifier, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # load a ResNet-18 model
        num_features = self.model.fc.in_features  # get the number of features in the last layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),  # dropout with 50% probability
            nn.Linear(num_features, num_classes)  # fully connected layer for classification
        )

    def forward(self, x):
        return self.model(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cuda":
    print("Running on GPU.")
else:
    print("Running on CPU.")

data = "C:/Users/benja/Desktop/DiabeticRetinopathy/archive/gaussian_filtered_images/gaussian_filtered_images"
split_dataset(data, 10)

train_dir = os.path.join(data, 'train')
val_dir = os.path.join(data, 'val')
test_dir = os.path.join(data, 'test')

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ensure images are 224x224 pixels
    transforms.ToTensor(),  # convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # imageNet normalization
])

train_dataset = CustomImageDataset(image_folder=train_dir, transform=transform, preprocess=preprocess_image_cv2)
val_dataset = CustomImageDataset(image_folder=val_dir, transform=transform, preprocess=preprocess_image_cv2)
test_dataset = CustomImageDataset(image_folder=test_dir, transform=transform, preprocess=preprocess_image_cv2)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = ResNetClassifier(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

early_stopper = EarlyStopping(patience=5, verbose=True, save_path='best_model.pth')

print("\033[1;31mBeginning training...\033[0m")
for epoch in range(15):
    start_time = time.time()

    # training phase
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        train_loss += loss.item()

    train_accuracy = correct_train / total_train
    train_loss = train_loss / len(train_loader.dataset)

    # validation phase
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            val_loss += loss.item()

    val_accuracy = val_correct / val_total
    val_loss = val_loss / len(val_loader.dataset)
    epoch_duration = time.time() - start_time
    print(f"Epoch {epoch+1}, Train Accuracy: {train_accuracy:.2f}, Train Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}, Validation Loss: {val_loss:.4f}, Time: {epoch_duration:.2f} seconds")

    early_stopper(val_loss, model)
    if early_stopper.early_stop:
        print("\033[1;31mEarly stopping triggered.\033[0m")
        break

print("\033[1;32mFinished training.\033[0m")

checkpoint = torch.load('best_model.pth', weights_only=True)  # specify weights_only=True
model.load_state_dict(checkpoint)

model.eval()  # set the model to evaluation mode
test_correct = 0
test_total = 0
test_loss = 0.0
start_time = time.time()
print("\033[1;31mBeginning testing...\033[0m")
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # track accuracy
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

        test_loss += loss.item()

test_loss = test_loss / len(test_loader.dataset)

accuracy = test_correct / test_total * 100
test_duration = time.time() - start_time
print(f'Test Loss: {test_loss:.6f}, Test Accuracy: {accuracy:.2f}%, Time: {test_duration:.2f} seconds')
print("\033[1;32mFinished testing.\033[0m")
