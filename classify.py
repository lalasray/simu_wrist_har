import torch
import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # Define your convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Define your max pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Define your fully connected layers
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 11)  # 11 classes
    
    def forward(self, x):
        # Apply convolutional layers with ReLU activation function and pooling
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        
        # Flatten the input for fully connected layers
        x = x.view(-1, 128 * 32 * 32)
        
        # Apply fully connected layers with ReLU activation function
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage:
# Instantiate the classifier
classifier = CNNClassifier()

# Generate a random input batch
input_batch = torch.randn(10, 3, 256)  # Batch size of 10

# Make predictions
predictions = classifier(input_batch)
print(predictions.shape)
# You can now use predictions for further processing, like calculating loss and backpropagation in a training loop.
