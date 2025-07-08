import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import get_model
from dataset import load_mnist_data

def prepare_data(train_images, train_labels, test_images, test_labels, batch_size=64):
    """Convert numpy arrays to PyTorch DataLoaders"""
    # Convert to tensors
    train_x = torch.from_numpy(train_images).float() / 255.0  # Normalize pixels
    train_y = torch.from_numpy(train_labels).long() 
    test_x = torch.from_numpy(test_images).float() / 255.0
    test_y = torch.from_numpy(test_labels).long()
    
    # Create datasets
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader



def train_model(model, train_loader, test_loader, epochs=5, lr=0.001):
    """Train the model"""
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # For classification
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            # Print progress
            if batch_idx % 200 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Calculate training accuracy
        train_accuracy = 100 * correct_train / total_train
        avg_loss = running_loss / len(train_loader)
        
        # Test phase
        test_accuracy = evaluate_model(model, test_loader)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'  Test Accuracy: {test_accuracy:.2f}%')
        print('-' * 50)

def evaluate_model(model, test_loader):
    """Evaluate model on test data"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def main():
    """Main training function"""
    print("=== MNIST Training ===")
    
    # Load data
    print("Loading MNIST data...")
    train_data, test_data = load_mnist_data("../data")
    train_images, train_labels = train_data
    test_images, test_labels = test_data
    
    if train_images is None or test_images is None:
        print("Error: Could not load data!")
        return
    
    print(f"Train data: {train_images.shape}")
    print(f"Test data: {test_images.shape}")
    
    # Prepare data loaders
    train_loader, test_loader = prepare_data(train_images, train_labels, 
                                           test_images, test_labels, batch_size=64)
    
    # Create model
    model = get_model("simple")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    train_model(model, train_loader, test_loader, epochs=5, lr=0.001)
    
    # Save model
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("Model saved as 'mnist_model.pth'")

if __name__ == "__main__":
    main()