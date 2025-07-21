#  MNIST Handwritten Digit Recognition

A PyTorch implementation of neural networks for classifying handwritten digits (0-9) using the MNIST dataset.

##  Project Overview

This project implements and compares two different neural network architectures:
- **Simple Neural Network**: Fully connected layers (97.35% accuracy)
- **Convolutional Neural Network**: Specialized for image recognition (99%+ accuracy)

##  Results

### Simple Neural Network Performance:
- **Global Accuracy**: 97.35%
- **Parameters**: 109,386
- **Training Time**: ~2 minutes (5 epochs)

### Per-Digit Accuracy:
```
Digit | Accuracy
------|----------
  0   |  98.67%
  1   |  99.21%
  2   |  96.90%
  3   |  97.52%
  4   |  97.15%
  5   |  96.64%
  6   |  98.33%
  7   |  97.37%
  8   |  95.68%
  9   |  96.53%
```

##  Project Structure

```
mnist/
├── data/                          # MNIST dataset files
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── t10k-images.idx3-ubyte
│   └── t10k-labels.idx1-ubyte
├── src/                           # Source code
│   ├── __init__.py
│   ├── dataset.py                 # MNIST data parser
│   ├── model.py                   # Neural network architectures
│   └── train.py                   # Training script
├── test_model.py                  # Model evaluation
├── main.py                        # Main entry point
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

##  Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Ozheim/Mnist-.git
cd mnist
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download MNIST data
Place the MNIST `.idx-ubyte` files in the `data/` folder:
- `train-images.idx3-ubyte` (60,000 training images)
- `train-labels.idx1-ubyte` (60,000 training labels)
- `t10k-images.idx3-ubyte` (10,000 test images)
- `t10k-labels.idx1-ubyte` (10,000 test labels)

### 4. Train the model
```bash
cd src
python train.py
```

### 5. Evaluate the model
```bash
python test_model.py
```

##  Architecture

### Simple Neural Network (MNISTNet)
```
Input (784) → Dense(128) → ReLU → Dropout(0.2) 
            → Dense(64)  → ReLU → Dropout(0.2)
            → Dense(10)  → Output
```

### Convolutional Neural Network (MNISTConvNet)
```
Input (28×28×1) → Conv2d(32) → ReLU → MaxPool
                → Conv2d(64) → ReLU → MaxPool
                → Dense(128) → ReLU → Dropout(0.5)
                → Dense(10)  → Output
```

## Training Details

- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 64
- **Epochs**: 5
- **Data Augmentation**: Normalization (pixel values ÷ 255)

##  Usage Examples

### Train a model
```python
from src.model import get_model
from src.train import train_model

# Create and train simple model
model = get_model("simple")
train_model(model, train_loader, test_loader, epochs=5)
```

### Evaluate model
```python
from src.model import get_model
import torch

# Load trained model
model = get_model("simple")
model.load_state_dict(torch.load('mnist_model.pth'))

# Make prediction
prediction = model(image_tensor)
predicted_digit = torch.argmax(prediction, dim=1)
```

##  Requirements

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
matplotlib>=3.5.0
```

##  Key Features

-  **Custom MNIST parser**: Reads binary `.idx-ubyte` format
-  **Two model architectures**: Simple NN vs CNN comparison
-  **Comprehensive evaluation**: Global and per-digit accuracy
-  **Clean code structure**: Modular and extensible
-  **GPU support**: Automatic CUDA detection
-  **Model persistence**: Save/load trained weights

##  Model Analysis

### Confusion Analysis
The model performs best on:
- **Digit 1**: 99.21% (simple vertical lines)
- **Digit 0**: 98.67% (clear circular shape)

The model struggles most with:
- **Digit 8**: 95.68% (complex curves, similar to 6/9)
- **Digit 9**: 96.53% (can be confused with 4/7)

### Confidence Levels
- **High confidence (>99%)**: Clear, well-written digits
- **Medium confidence (90-99%)**: Slightly unclear handwriting
- **Low confidence (<90%)**: Ambiguous or poorly written digits

##  Future Improvements

- [ ] Data augmentation (rotation, scaling)
- [ ] Learning rate scheduling
- [ ] Ensemble methods
- [ ] Model compression techniques
- [ ] Real-time digit recognition interface

##  Comparison with State-of-the-Art

| Model Type | Accuracy | Parameters | Training Time |
|------------|----------|------------|---------------|
| Simple NN  | 97.35%   | 109K       | 2 min         |
| CNN        | 99%+     | ~50K       | 3 min         |
| ResNet     | 99.5%+   | 1M+        | 10+ min       |

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgments

- **MNIST Dataset**: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- **PyTorch**: Facebook AI Research team
- **Inspiration**: Classic machine learning education
