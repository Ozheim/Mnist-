import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import numpy as np
from PIL import Image, ImageDraw
import torch
import sys
import os
import time

# Ajouter le dossier src au path
sys.path.append('src')
from model import get_model

class MNISTCollectApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MNIST with Feedback Learning")
        self.root.geometry("500x800")
        
        # Variables
        self.canvas_size = 280
        self.brush_size = 12  # Plus petit pour mieux ressembler à MNIST
        self.drawing = False
        
        # Feedback tracking
        self.corrections_count = 0
        self.current_prediction = None
        self.current_confidence = None
        
        # Counter for saved images (BEFORE setup_ui!)
        self.saved_count = 0
        
        # Create data collection folders
        self.create_data_folders()
        
        # Load trained model (priorité au modèle personnel)
        self.model = None
        self.load_model()
        
        # Create UI
        self.setup_ui()
        
        # Canvas for PIL drawing
        self.pil_image = Image.new('L', (self.canvas_size, self.canvas_size), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        
    def create_data_folders(self):
        """Create folders for data collection"""
        self.data_dir = "mes_donnees"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        # Create folders for each digit
        for i in range(10):
            digit_folder = os.path.join(self.data_dir, str(i))
            if not os.path.exists(digit_folder):
                os.makedirs(digit_folder)
                
        print(f"Data collection folders created in: {self.data_dir}")
        
    def load_model(self):
        """Load the trained MNIST model (personal model ONLY)"""
        model_paths = [
            'personal_mnist_model.pth',  # Modèle personnel uniquement
        ]
        
        self.model = get_model("simple")
        
        for path in model_paths:
            try:
                if os.path.exists(path):
                    self.model.load_state_dict(torch.load(path))
                    self.model.eval()
                    print(f"Personal model loaded successfully from: {path}")
                    return
            except Exception as e:
                print(f"Failed to load from {path}: {e}")
        
        print("Could not load personal model!")
        self.model = None
    
    def setup_ui(self):
        """Create the user interface"""
        # Title
        title_label = tk.Label(self.root, text="MNIST with Feedback Learning", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Drawing canvas
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, 
                               height=self.canvas_size, bg='black', 
                               cursor="pencil")
        self.canvas.pack(pady=10)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # Buttons frame 1
        button_frame1 = tk.Frame(self.root)
        button_frame1.pack(pady=5)
        
        # Predict button
        self.predict_btn = tk.Button(button_frame1, text="Predict", 
                                    command=self.predict_digit,
                                    font=("Arial", 12, "bold"),
                                    bg="blue", fg="white",
                                    width=10)
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        self.clear_btn = tk.Button(button_frame1, text="Clear", 
                                  command=self.clear_canvas,
                                  font=("Arial", 12, "bold"),
                                  bg="red", fg="white",
                                  width=10)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Result frame
        result_frame = tk.Frame(self.root)
        result_frame.pack(pady=10)
        
        # Prediction result
        self.result_label = tk.Label(result_frame, text="Draw a digit and click Predict", 
                                    font=("Arial", 14))
        self.result_label.pack()
        
        # Confidence
        self.confidence_label = tk.Label(result_frame, text="", 
                                        font=("Arial", 12))
        self.confidence_label.pack()
        
        # FEEDBACK SECTION (NEW!)
        feedback_frame = tk.LabelFrame(self.root, text="Feedback", 
                                      font=("Arial", 12, "bold"))
        feedback_frame.pack(pady=15, padx=20, fill="x")
        
        # Feedback buttons
        feedback_buttons_frame = tk.Frame(feedback_frame)
        feedback_buttons_frame.pack(pady=10)
        
        self.correct_btn = tk.Button(feedback_buttons_frame, text="✅ Correct", 
                                    command=self.mark_correct,
                                    font=("Arial", 11, "bold"),
                                    bg="green", fg="white",
                                    width=12, state="disabled")
        self.correct_btn.pack(side=tk.LEFT, padx=5)
        
        self.wrong_btn = tk.Button(feedback_buttons_frame, text="❌ Wrong", 
                                  command=self.mark_wrong,
                                  font=("Arial", 11, "bold"),
                                  bg="orange", fg="white",
                                  width=12, state="disabled")
        self.wrong_btn.pack(side=tk.LEFT, padx=5)
        
        # Correction buttons (0-9)
        tk.Label(feedback_frame, text="If wrong, click the correct digit:", 
                font=("Arial", 10)).pack(pady=(10,5))
        
        correction_frame = tk.Frame(feedback_frame)
        correction_frame.pack()
        
        self.correction_buttons = []
        for i in range(10):
            btn = tk.Button(correction_frame, text=str(i), 
                           command=lambda digit=i: self.correct_to(digit),
                           font=("Arial", 10), width=3, height=1,
                           state="disabled")
            btn.pack(side=tk.LEFT, padx=2)
            self.correction_buttons.append(btn)
        
        # Quick save buttons frame
        quick_frame = tk.LabelFrame(self.root, text="Quick Save (no prediction needed)")
        quick_frame.pack(pady=10, padx=20, fill="x")

        tk.Label(quick_frame, text="Draw and save directly:", font=("Arial", 10)).pack()
        
        # Quick save buttons (0-9)
        buttons_frame = tk.Frame(quick_frame)
        buttons_frame.pack(pady=5)
        
        for i in range(10):
            btn = tk.Button(buttons_frame, text=str(i), 
                           command=lambda digit=i: self.quick_save(digit),
                           font=("Arial", 10), width=3, height=1)
            btn.pack(side=tk.LEFT, padx=2)
        
        # Collection stats
        stats_frame = tk.Frame(self.root)
        stats_frame.pack(pady=10)
        
        self.stats_label = tk.Label(stats_frame, text=f"Images saved: {self.saved_count}", 
                                   font=("Arial", 10))
        self.stats_label.pack()
        
        self.corrections_label = tk.Label(stats_frame, text=f"Corrections: {self.corrections_count}", 
                                         font=("Arial", 10), fg="blue")
        self.corrections_label.pack()
        
        # Re-train button (appears when enough corrections)
        self.retrain_btn = tk.Button(stats_frame, text="🔄 Re-train Model", 
                                    command=self.retrain_model,
                                    font=("Arial", 10, "bold"),
                                    bg="purple", fg="white",
                                    state="disabled")
        self.retrain_btn.pack(pady=5)
        
        # Status
        self.status_label = tk.Label(self.root, 
                                    text="Personal model loaded" if self.model else "Personal model not loaded", 
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
            
            # Draw on PIL image (for prediction and saving)
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
        self.disable_feedback_buttons()
    
    def predict_digit(self):
        """Predict the drawn digit"""
        if not self.model:
            self.result_label.config(text="Personal model not loaded!", fg="red")
            return
        
        if self.is_canvas_empty():
            self.result_label.config(text="Please draw something first!", fg="orange")
            return
        
        try:
            # Resize image to 28x28
            img_resized = self.pil_image.resize((28, 28), Image.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            
            # Convert to tensor and add batch dimension
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_digit = torch.argmax(output, dim=1).item()
                confidence = probabilities[0][predicted_digit].item() * 100
            
            # Store prediction for feedback
            self.current_prediction = predicted_digit
            self.current_confidence = confidence
            
            # Display result
            self.result_label.config(text=f"Predicted digit: {predicted_digit}", 
                                    fg="blue", font=("Arial", 16, "bold"))
            self.confidence_label.config(text=f"Confidence: {confidence:.1f}%", 
                                        fg="green" if confidence > 90 else "orange")
            
            # Enable feedback buttons
            self.enable_feedback_buttons()
            
        except Exception as e:
            self.result_label.config(text=f"Prediction error: {str(e)}", fg="red")
    
    def enable_feedback_buttons(self):
        """Enable feedback buttons after prediction"""
        self.correct_btn.config(state="normal")
        self.wrong_btn.config(state="normal")
        for btn in self.correction_buttons:
            btn.config(state="normal")
    
    def disable_feedback_buttons(self):
        """Disable feedback buttons"""
        self.correct_btn.config(state="disabled")
        self.wrong_btn.config(state="disabled")
        for btn in self.correction_buttons:
            btn.config(state="disabled")
    
    def mark_correct(self):
        """Mark prediction as correct and save"""
        if self.current_prediction is not None:
            self.save_image(self.current_prediction)
            self.result_label.config(text=f"✅ Correct! Saved as {self.current_prediction}", fg="green")
            self.disable_feedback_buttons()
    
    def mark_wrong(self):
        """Mark prediction as wrong"""
        self.result_label.config(text="❌ Wrong! Click the correct digit below:", fg="red")
        self.wrong_btn.config(state="disabled")
        self.correct_btn.config(state="disabled")
    
    def correct_to(self, correct_digit):
        """Correct prediction to the right digit"""
        if self.current_prediction is not None:
            self.save_image(correct_digit)
            self.corrections_count += 1
            self.corrections_label.config(text=f"Corrections: {self.corrections_count}")
            
            # Show correction feedback
            self.result_label.config(text=f"✅ Corrected! Was {self.current_prediction}, now saved as {correct_digit}", 
                                    fg="green")
            
            # Check if we should enable retrain button
            if self.corrections_count >= 1:  # After 1 correction (pour test)
                self.retrain_btn.config(state="normal")
            
            self.disable_feedback_buttons()
    
    def retrain_model(self):
        """Launch retraining process"""
        result = messagebox.askyesno("Re-train Model", 
                                    f"You have {self.corrections_count} corrections.\n"
                                    f"Do you want to re-train the model with new data?\n"
                                    f"This may take a few minutes...")
        
        if result:
            self.result_label.config(text="🔄 Re-training model...", fg="blue")
            self.retrain_btn.config(state="disabled", text="Training...")
            self.root.update()
            
            try:
                # Launch retrain script
                import subprocess
                result = subprocess.run(['python', 'retrain.py'], 
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Reload the new model
                    self.load_model()
                    self.result_label.config(text="✅ Model re-trained successfully!", fg="green")
                    self.corrections_count = 0  # Reset counter
                    self.corrections_label.config(text=f"Corrections: {self.corrections_count}")
                    self.retrain_btn.config(state="disabled")  # Disable again
                else:
                    self.result_label.config(text="❌ Re-training failed", fg="red")
                    
            except Exception as e:
                self.result_label.config(text=f"❌ Re-training error: {str(e)}", fg="red")
            
            self.retrain_btn.config(text="🔄 Re-train Model")
    
    def quick_save(self, digit):
        """Quick save with predetermined digit"""
        if self.is_canvas_empty():
            messagebox.showwarning("Warning", "Please draw something first!")
            return
        
        self.save_image(digit)
        self.result_label.config(text=f"Saved as digit {digit}!", fg="green")
    
    def save_image(self, digit):
        """Save image to the appropriate folder"""
        try:
            # Resize image to 28x28 (same as MNIST)
            img_resized = self.pil_image.resize((28, 28), Image.LANCZOS)
            
            # Create filename with timestamp
            timestamp = int(time.time() * 1000)  # milliseconds
            filename = f"digit_{digit}_{timestamp}.png"
            filepath = os.path.join(self.data_dir, str(digit), filename)
            
            # Save image
            img_resized.save(filepath)
            
            # Update counter and stats
            self.saved_count += 1
            self.stats_label.config(text=f"Images saved: {self.saved_count}")
            
            print(f"Saved image: {filepath}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {str(e)}")
    
    def is_canvas_empty(self):
        """Check if canvas is empty"""
        img_array = np.array(self.pil_image)
        return img_array.max() == 0
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

def main():
    """Main function"""
    print("Starting MNIST Feedback Learning Application...")
    print("Instructions:")
    print("1. Draw a digit (0-9) on the black canvas")
    print("2. Click 'Predict' to see the prediction")
    print("3. Give feedback:")
    print("   - Click '✅ Correct' if prediction is right")
    print("   - Click '❌ Wrong' then the correct digit if wrong")
    print("4. After 1+ corrections, you can re-train the model!")
    print("5. Use quick save buttons for direct saving")
    print("6. Use 'Clear' to start over")
    print()
    
    app = MNISTCollectApp()
    app.run()

if __name__ == "__main__":
    main()