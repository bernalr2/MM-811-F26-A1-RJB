# Assignment 1: LeNet Image Classification Network
# MM 811 F26
# Ryan James Bernal | University of Alberta | 1751934

# Purpose: This file creates a LeNet Image Classification Network trained using the MNIST Dataset

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# Create the LeNet Image Classification Network Class
"""
Purpose: This class builds the framework for the LeNet Image Classification Network
Parameters: None
Pre: None
Post: None
Return: Network(), an object of the LeNet Image Classification Network
Raise: None
"""
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=256, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    """
    Purpose: Processes the data through layers of the LeNet Image Classification Network
    Parameters: x, the images being processed
    Pre: x, the images being processed
    Post: x, the data from the images being used for classification
    Return: x, the data after running the image data through the network
    Raise: None
    """
    def forward(self, x):
        # First Convolution and Subsample
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Second Convolution and Subsample
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)

        # Perform Full Connections
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x

"""
Purpose: Download and ready the MNIST Dataset for training and testing
Parameters: None
Pre: None
Post: The MNIST Dataset for training and testing becomes ready
Return: train_loader, test_loader - a tuple pair containing the corresponding images
Raise: None
"""
def load_dataset():
    print("==========MNIST Training and Test Data Loading...==========")
    transform = transforms.ToTensor()

    # Download the Training Data
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    # Download the Test Data
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Load the Training Data
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Load the Test Data
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("==========MNIST Training and Test Data Successfully Loaded!==========")
    return train_loader, test_loader

"""
Purpose: Train the model and test it to check for its label accuracy
Parameters: None
Pre: None
Post: The model and its results after training and testing will be posted
Return: None
Raise: None
"""
def main():
    print("**********Start of Program**********")
    # Load the MNIST Dataset
    train_loader, test_loader = load_dataset()

    # Find device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Compile and set the model
    model = Network().to(device)
    criterion = (nn.CrossEntropyLoss())
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train The Model
    print("==========Model Training==========")
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Use the images and labels in the training loader to train the model
        for images, labels in train_loader:

            # Forward Pass
            outputs = model(images)

            # Calculate Prediction Error
            loss = criterion(outputs, labels)

            # Backward Pass: Calculate Gradients
            optimizer.zero_grad()
            loss.backward()

            # Update model weights
            optimizer.step()

            # Save current training progress
            running_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate current loss and model accuracy
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss: {epoch_loss:.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )

    print("==========Model Evaluation==========")
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        # Test the Model and Demonstrate Output Results
        for images, labels in test_loader:
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100

    print(f"Test Accuracy: {accuracy:.2f}%")

    print("**********End of Program**********")

# Main Function Code
if __name__ == "__main__":
    main()