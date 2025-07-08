import sys
sys.path.append('src')

import torch
from model import get_model
from dataset import load_mnist_data

def test_model_basic():
    """Test if model works with dummy data"""
    print("=== Testing model with dummy data ===")
    
    # Create model
    model = get_model("simple")
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create fake data (batch of 5 images)
    fake_images = torch.randn(5, 28, 28)
    print(f"Fake input shape: {fake_images.shape}")
    
    # Forward pass (prediction)
    with torch.no_grad():  # Don't calculate gradients for testing
        output = model(fake_images)
    
    print(f"Output shape: {output.shape}")
    print(f"Output (raw scores): {output[0]}")
    
    # Convert to probabilities
    probabilities = torch.softmax(output, dim=1)
    print(f"Probabilities: {probabilities[0]}")
    
    # Get predictions (highest score)
    predictions = torch.argmax(output, dim=1)
    print(f"Predictions: {predictions}")

def test_model_real_data():
    """Test model with real MNIST data"""
    print("\n=== Testing model with real MNIST data ===")
    
    # Load data
    try:
        train_data, test_data = load_mnist_data("../data")
        test_images, test_labels = test_data
        
        if test_images is None:
            print("No test images found!")
            return
        
        # Convert to PyTorch tensors and normalize
        images_tensor = torch.from_numpy(test_images).float() / 255.0
        labels_tensor = torch.from_numpy(test_labels).long()
        
        # Create model
        model = get_model("simple")
        
        # Try to load trained weights
        try:
            model.load_state_dict(torch.load('mnist_model.pth'))
            model.eval()
        except:
            print("⚠️  No trained model found, using random weights")
        
        # Make predictions in batches
        all_predictions = []
        batch_size = 1000
        
        with torch.no_grad():
            for i in range(0, len(images_tensor), batch_size):
                batch = images_tensor[i:i+batch_size]
                output = model(batch)
                predictions = torch.argmax(output, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
        
        all_predictions = torch.tensor(all_predictions)
        
        # Global accuracy
        correct_total = (all_predictions == labels_tensor).sum().item()
        global_accuracy = correct_total / len(labels_tensor) * 100
        
        print(f"GLOBAL ACCURACY: {correct_total}/{len(labels_tensor)} = {global_accuracy:.2f}%")
        
        # Accuracy per digit (0-9)
        print(f"\nACCURACY PER DIGIT:")
        print("Digit | Correct | Total | Accuracy")
        print("-" * 35)
        
        for digit in range(10):
            # Find all instances of this digit
            digit_mask = (labels_tensor == digit)
            digit_total = digit_mask.sum().item()
            
            if digit_total > 0:
                # Check predictions for this digit
                digit_predictions = all_predictions[digit_mask]
                digit_labels = labels_tensor[digit_mask]
                digit_correct = (digit_predictions == digit_labels).sum().item()
                digit_accuracy = digit_correct / digit_total * 100
                
                print(f"  {digit}   |  {digit_correct:4d}   | {digit_total:4d}  |  {digit_accuracy:6.2f}%")
            else:
                print(f"  {digit}   |   0    |   0   |    N/A")
        
    except Exception as e:
        print(f"Error loading data: {e}")

if __name__ == "__main__":
    test_model_basic()
    test_model_real_data()