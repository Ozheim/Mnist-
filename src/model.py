import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    """Simple neural network for MNIST classification"""
    
    def __init__(self):
        super(MNISTNet, self).__init__()
        # Input: 28x28 = 784 pixels
        self.fc1 = nn.Linear(784, 128)    # First hidden layer
        self.fc2 = nn.Linear(128, 64)     # Second hidden layer  
        self.fc3 = nn.Linear(64, 10)      # Output layer (10 classes)
        self.dropout = nn.Dropout(0.2)    # Prevent overfitting
        
    def forward(self, x):
        # Flatten image from 28x28 to 784
        x = x.view(-1, 784)
        
        # Hidden layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer (no activation, we'll use CrossEntropyLoss)
        x = self.fc3(x)
        return x

class MNISTConvNet(nn.Module):
    """Convolutional neural network for MNIST"""
    
    def __init__(self):
        super(MNISTConvNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2, 2)                           # 28x28 -> 14x14
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # After 2 pooling: 28->14->7
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Add channel dimension: (batch, 28, 28) -> (batch, 1, 28, 28)
        x = x.unsqueeze(1)
        
        # Conv layers
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        
        # Flatten for FC layers
        x = x.view(-1, 64 * 7 * 7)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_model(model_type="simple"):
    """Get model instance"""
    if model_type == "simple":
        return MNISTNet()
    elif model_type == "conv":
        return MNISTConvNet()
    else:
        raise ValueError("Model type must be 'simple' or 'conv'")

if __name__ == "__main__":
    # Test models
    print("=== Testing Models ===")
    
    # Test simple model
    model_simple = get_model("simple")
    print(f"Simple model: {sum(p.numel() for p in model_simple.parameters())} parameters")
    
    # Test conv model
    model_conv = get_model("conv")
    print(f"Conv model: {sum(p.numel() for p in model_conv.parameters())} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(32, 28, 28)  # Batch of 32 images
    
    output_simple = model_simple(dummy_input)
    output_conv = model_conv(dummy_input)
    
    print(f"Simple model output shape: {output_simple.shape}")
    print(f"Conv model output shape: {output_conv.shape}")