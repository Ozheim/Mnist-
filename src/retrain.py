import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import sys

# Ajouter le dossier src au path
sys.path.append('src')
from model import get_model

class PersonalMNISTDataset(Dataset):
    """Dataset for personal MNIST data"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.images = []
        self.labels = []
        
        # Load all images from folders 0-9
        for digit in range(10):
            digit_folder = os.path.join(data_dir, str(digit))
            if os.path.exists(digit_folder):
                for filename in os.listdir(digit_folder):
                    if filename.endswith('.png'):
                        img_path = os.path.join(digit_folder, filename)
                        self.images.append(img_path)
                        self.labels.append(digit)
        
        print(f"Loaded {len(self.images)} personal images")
        self.print_distribution()
    
    def print_distribution(self):
        """Print distribution of digits"""
        counts = [0] * 10
        for label in self.labels:
            counts[label] += 1
        
        print("Distribution of personal data:")
        for digit, count in enumerate(counts):
            print(f"  Digit {digit}: {count} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')  # Grayscale
        
        # Convert to tensor and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array)
        
        # Label
        label = self.labels[idx]
        
        return img_tensor, label

def test_model_on_personal_data(model, test_loader):
    """Test model accuracy on personal data"""
    model.eval()
    correct = 0
    total = 0
    per_digit_correct = [0] * 10
    per_digit_total = [0] * 10
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per digit accuracy
            for i in range(labels.size(0)):
                digit = labels[i].item()
                per_digit_total[digit] += 1
                if predicted[i] == labels[i]:
                    per_digit_correct[digit] += 1
    
    accuracy = 100 * correct / total if total > 0 else 0
    
    print(f"Overall accuracy: {correct}/{total} = {accuracy:.2f}%")
    print("Per-digit accuracy:")
    for digit in range(10):
        if per_digit_total[digit] > 0:
            digit_acc = 100 * per_digit_correct[digit] / per_digit_total[digit]
            print(f"  Digit {digit}: {per_digit_correct[digit]}/{per_digit_total[digit]} = {digit_acc:.2f}%")
    
    return accuracy

def fine_tune_model():
    """Fine-tune the model on personal data"""
    
    print("=== MNIST Transfer Learning ===")
    
    # Check if personal data exists
    data_dir = "mes_donnees"
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} folder not found!")
        print("Please collect some personal data first using draw_collect.py")
        return
    
    # Load personal dataset
    personal_dataset = PersonalMNISTDataset(data_dir)
    
    if len(personal_dataset) == 0:
        print("No personal data found!")
        print("Please draw and save some digits first.")
        return
    
    # Split data (80% train, 20% test)
    dataset_size = len(personal_dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        personal_dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"Training on {train_size} images, testing on {test_size} images")
    
    # Load pre-trained model
    model = get_model("simple")
    
    try:
        model.load_state_dict(torch.load('src/mnist_model.pth'))
        print("✅ Pre-trained MNIST model loaded")
    except:
        print("❌ Could not load pre-trained model")
        print("Training from scratch...")
    
    # Test model BEFORE fine-tuning
    print("\n=== BEFORE Fine-tuning ===")
    initial_accuracy = test_model_on_personal_data(model, test_loader)
    
    # Fine-tuning setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Higher learning rate
    
    # Fine-tuning
    print(f"\n=== Starting Fine-tuning ===")
    epochs = 20  # More epochs
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        train_accuracy = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{epochs}: Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
    
    # Test model AFTER fine-tuning
    print("\n=== AFTER Fine-tuning ===")
    final_accuracy = test_model_on_personal_data(model, test_loader)
    
    # Compare results
    print(f"\n=== RESULTS COMPARISON ===")
    print(f"Before fine-tuning: {initial_accuracy:.2f}%")
    print(f"After fine-tuning:  {final_accuracy:.2f}%")
    improvement = final_accuracy - initial_accuracy
    print(f"Improvement: {improvement:+.2f}%")
    
    # Save fine-tuned model
    torch.save(model.state_dict(), 'personal_mnist_model.pth')
    print("\n✅ Fine-tuned model saved as 'personal_mnist_model.pth'")
    
    return model

def main():
    """Main function"""
    print("Personal MNIST Transfer Learning")
    print("=" * 40)
    
    # Check data
    data_dir = "mes_donnees"
    if not os.path.exists(data_dir):
        print("❌ No personal data found!")
        print("Please run draw_collect.py first to collect your drawings.")
        return
    
    # Count images per digit
    total_images = 0
    for digit in range(10):
        digit_folder = os.path.join(data_dir, str(digit))
        if os.path.exists(digit_folder):
            count = len([f for f in os.listdir(digit_folder) if f.endswith('.png')])
            total_images += count
            print(f"Digit {digit}: {count} images")
    
    print(f"\nTotal personal images: {total_images}")
    
    if total_images < 10:
        print("⚠️  You need more data for good results!")
        print("Recommendation: At least 10-20 images per digit")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Start fine-tuning
    fine_tune_model()

if __name__ == "__main__":
    main()