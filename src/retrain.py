import os, torch, torch.nn as nn, torch.optim as optim, sys
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

sys.path.append('src')
from model import get_model

class PersonalMNISTDataset(Dataset):
    def __init__(self, data_dir):
        self.images, self.labels = [], []
        
        for digit in range(10):
            folder = os.path.join(data_dir, str(digit))
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if f.endswith('.png'):
                        self.images.append(os.path.join(folder, f))
                        self.labels.append(digit)
        
        counts = [0] * 10
        for label in self.labels:
            counts[label] += 1
        
        print(f"Loaded {len(self.images)} images: {counts}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('L')
        return torch.from_numpy(np.array(img, dtype=np.float32) / 255.0), self.labels[idx]

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    per_digit = [0] * 10
    per_total = [0] * 10
    
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            for i in range(labels.size(0)):
                digit = labels[i].item()
                per_total[digit] += 1
                if predicted[i] == labels[i]:
                    per_digit[digit] += 1
    
    accuracy = 100 * correct / total
    print(f"Accuracy: {correct}/{total} = {accuracy:.2f}%")
    for i in range(10):
        if per_total[i] > 0:
            acc = 100 * per_digit[i] / per_total[i]
            print(f"  {i}: {per_digit[i]}/{per_total[i]} = {acc:.2f}%")
    
    return accuracy

def main():
    data_dir = "mes_donnees"
    if not os.path.exists(data_dir):
        print("❌ No data found! Run draw_collect.py first.")
        return
    
    # Load data
    dataset = PersonalMNISTDataset(data_dir)
    if len(dataset) < 10:
        print("⚠️  Need more data!")
        return
    
    # Split 80/20
    train_size = int(0.8 * len(dataset))
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)
    
    # Load model
    model = get_model("simple")
    try:
        model.load_state_dict(torch.load('src/mnist_model.pth'))
        print("✅ MNIST model loaded")
    except:
        print("❌ Using random weights")
    
    # Test before
    print("\n=== BEFORE ===")
    before = evaluate(model, test_loader)
    
    # Train
    print("\n=== TRAINING ===")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(20):
        loss_sum, correct, total = 0, 0, 0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item()
            correct += (torch.argmax(output, dim=1) == target).sum().item()
            total += target.size(0)
        
        if epoch % 5 == 0:  # Print every 5 epochs
            print(f"Epoch {epoch+1}: Loss={loss_sum/len(train_loader):.3f}, Acc={100*correct/total:.1f}%")
    
    # Test after
    print("\n=== AFTER ===")
    after = evaluate(model, test_loader)
    
    print(f"\n=== RESULTS ===")
    print(f"Before: {before:.2f}% → After: {after:.2f}% (Δ{after-before:+.2f}%)")
    
    # Save
    torch.save(model.state_dict(), 'personal_mnist_model.pth')
    print("✅ Model saved")

if __name__ == "__main__":
    main()