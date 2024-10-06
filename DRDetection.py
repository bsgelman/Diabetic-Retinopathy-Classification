import os
import random
import time
from shutil import copy2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def split_dataset(data_dir, seed=10):
    random.seed(seed)

    train_ratio = 0.64  # 64% for training
    val_ratio = 0.16  # 16% for validation
    test_ratio = 0.20  # 20% for testing

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    if os.path.exists(train_dir) or os.path.exists(val_dir) or os.path.exists(test_dir):
        print("WARNING: Training, validation, and/or testing directories already exist.")
        return
    os.makedirs(train_dir)
    os.makedirs(test_dir)
    os.makedirs(val_dir)

    for class_name in os.listdir(data_dir):  # Iterate through each severity folder
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir) and class_name not in ['train', 'val', 'test', 'Binary']:
            images = os.listdir(class_dir)
            random.shuffle(images)
            #images = images[:100]

            train_size = int(len(images) * train_ratio)
            val_size = int(len(images) * val_ratio)
            train_images = images[:train_size]
            val_images = images[train_size:train_size + val_size]
            test_images = images[train_size + val_size:]  # Remainder of images for testing

            # Create subdirectories for each class in train, val, and test folders
            train_class_dir = os.path.join(train_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)

            os.makedirs(train_class_dir)
            os.makedirs(val_class_dir)
            os.makedirs(test_class_dir)

            # Copy files to respective directories
            for img in train_images:
                copy2(os.path.join(class_dir, img), os.path.join(train_class_dir, img))
            for img in val_images:
                copy2(os.path.join(class_dir, img), os.path.join(val_class_dir, img))
            for img in test_images:
                copy2(os.path.join(class_dir, img), os.path.join(test_class_dir, img))

    print("\033[1;32mData split completed.\033[0m")


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),  # Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112x112

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),  # Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),  # Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14

            #nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # New layer
            #nn.BatchNorm2d(512),
            #nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully connected layers
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            #nn.Linear(1024, 512),
            #nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(512, 5),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x


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
    transforms.Resize((224, 224)),  # Ensure images are 224x224 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

train_dataset = ImageFolder(root=train_dir, transform=transform)
val_dataset = ImageFolder(root=val_dir, transform=transform)
test_dataset = ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4) # remove weight decay for optimal training accuracy

# Early stopping variables
early_stop_patience = 10
min_val_loss = float('inf')
epochs_no_improve = 0

print("\033[1;31mBeginning training...\033[0m")
for epoch in range(20):
    start_time = time.time()

    # Training phase
    model.train()  # Set the model to training mode
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track accuracy
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        train_loss += loss.item()

    train_accuracy = correct_train / total_train
    train_loss = train_loss / len(train_loader.dataset)

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Track accuracy
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            val_loss += loss.item()

    val_accuracy = val_correct / val_total
    val_loss = val_loss / len(val_loader.dataset)
    epoch_duration = time.time() - start_time
    print(f"Epoch {epoch+1}, Train Accuracy: {train_accuracy:.2f}, Train Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}, Validation Loss: {val_loss:.4f}, Time: {epoch_duration:.2f} seconds")
    
    # Early stopping check
    #if val_loss < min_val_loss:
        #min_val_loss = val_loss
        #epochs_no_improve = 0
    #else:
        #epochs_no_improve += 1
        #if epochs_no_improve >= early_stop_patience:
            #print("\033[1;31mEarly stopping triggered.\033[0m")
            #break

print("\033[1;32mFinished training.\033[0m")

model.eval()  # Set the model to evaluation mode
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

        # Track accuracy
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

        test_loss += loss.item()

# Calculate average losses
test_loss = test_loss / len(test_loader.dataset)

# Calculate accuracy
accuracy = test_correct / test_total * 100
test_duration = time.time() - start_time
print(f'Test Loss: {test_loss:.6f}, Test Accuracy: {accuracy:.2f}%, Time: {test_duration:.2f} seconds')
print("\033[1;32mFinished testing.\033[0m")