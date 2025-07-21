import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw
import torch
import sys
import os

# Ajouter le dossier src au path
sys.path.append('src')
from model import get_model

class MNISTDrawingApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MNIST Digit Recognition")
        self.root.geometry("500x600")
        
        # Variables
        self.canvas_size = 280
        self.brush_size = 12
        self.drawing = False
        
        # Load trained model
        self.model = None
        self.load_model()
        
        # Create UI
        self.setup_ui()
        
        # Canvas for PIL drawing (for model prediction)
        self.pil_image = Image.new('L', (self.canvas_size, self.canvas_size), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        
    def load_model(self):
        """Load the trained MNIST model"""
        model_paths = [
            'src/mnist_model.pth',      # Depuis la racine
            './src/mnist_model.pth',    # Depuis la racine avec ./
            'mnist_model.pth',          # Dans le même dossier
            '../src/mnist_model.pth'    # Depuis un sous-dossier
        ]
        
        self.model = get_model("simple")
        
        for path in model_paths:
            try:
                if os.path.exists(path):
                    self.model.load_state_dict(torch.load(path))
                    self.model.eval()
                    print(f"Model loaded successfully from: {path}")
                    return
            except Exception as e:
                print(f"Failed to load from {path}: {e}")
        
        print("Could not load model from any path!")
        self.model = None
    
    def setup_ui(self):
        """Create the user interface"""
        # Title
        title_label = tk.Label(self.root, text="Draw a digit (0-9)", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Drawing canvas
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, 
                               height=self.canvas_size, bg='black')
        self.canvas.pack(pady=10)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # Buttons frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        # Predict button
        self.predict_btn = tk.Button(button_frame, text="Predict", 
                                    command=self.predict_digit,
                                    font=("Arial", 12, "bold"),
                                    bg="green", fg="white",
                                    width=10)
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        self.clear_btn = tk.Button(button_frame, text="Clear", 
                                  command=self.clear_canvas,
                                  font=("Arial", 12, "bold"),
                                  bg="red", fg="white",
                                  width=10)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Result frame
        result_frame = tk.Frame(self.root)
        result_frame.pack(pady=20)
        
        # Prediction result
        self.result_label = tk.Label(result_frame, text="Draw a digit and click Predict", 
                                    font=("Arial", 14))
        self.result_label.pack()
        
        # Confidence
        self.confidence_label = tk.Label(result_frame, text="", 
                                        font=("Arial", 12))
        self.confidence_label.pack()
        
        # Status
        self.status_label = tk.Label(self.root, 
                                    text="Model loaded" if self.model else "Model not loaded", 
                                    font=("Arial", 10),
                                    fg="green" if self.model else "red")
        self.status_label.pack(side=tk.BOTTOM, pady=5)
    
    def start_draw(self, event):
        """Start drawing"""
        self.drawing = True
        
    def draw(self, event):
        """Draw on canvas"""
        if self.drawing:
            x, y = event.x, event.y
            # Draw on tkinter canvas (visual)
            self.canvas.create_oval(x - self.brush_size//2, y - self.brush_size//2,
                                   x + self.brush_size//2, y + self.brush_size//2,
                                   fill='white', outline='white')
            
            # Draw on PIL image (for prediction)
            self.pil_draw.ellipse([x - self.brush_size//2, y - self.brush_size//2,
                                  x + self.brush_size//2, y + self.brush_size//2],
                                 fill=255)
    
    def stop_draw(self, event):
        """Stop drawing"""
        self.drawing = False
    
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.delete("all")
        self.pil_image = Image.new('L', (self.canvas_size, self.canvas_size), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        self.result_label.config(text="Draw a digit and click Predict")
        self.confidence_label.config(text="")
    
    def predict_digit(self):
        """Predict the drawn digit"""
        if not self.model:
            self.result_label.config(text="Model not loaded!", fg="red")
            return
        
        try:
            # Resize image to 28x28
            img_resized = self.pil_image.resize((28, 28), Image.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            
            # Check if image is empty
            if img_array.max() == 0:
                self.result_label.config(text="Please draw something first!", fg="orange")
                return
            
            # Convert to tensor and add batch dimension
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_digit = torch.argmax(output, dim=1).item()
                confidence = probabilities[0][predicted_digit].item() * 100
            
            # Display result
            self.result_label.config(text=f"Predicted digit: {predicted_digit}", 
                                    fg="blue", font=("Arial", 16, "bold"))
            self.confidence_label.config(text=f"Confidence: {confidence:.1f}%", 
                                        fg="green" if confidence > 90 else "orange")
            
        except Exception as e:
            self.result_label.config(text=f"Prediction error: {str(e)}", fg="red")
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

def main():
    """Main function"""
    print("Starting MNIST Drawing Application...")
    print("Instructions:")
    print("1. Draw a digit (0-9) on the black canvas")
    print("2. Click 'Predict' to see the prediction")
    print("3. Click 'Clear' to start over")
    print()
    
    app = MNISTDrawingApp()
    app.run()

if __name__ == "__main__":
    main()