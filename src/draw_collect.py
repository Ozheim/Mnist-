import tkinter as tk, os, time, sys, subprocess
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.append('src')
from model import get_model

class MNISTApp:
    def __init__(self):
        # Setup window
        self.root = tk.Tk()
        self.root.title("MNIST Feedback Learning")
        self.root.geometry("500x750")
        
        # Variables
        self.canvas_size, self.brush_size = 280, 12
        self.drawing = False
        self.saved_count = self.corrections_count = 0
        self.current_prediction = self.current_confidence = None
        
        # Create folders and load model
        self.data_dir = "mes_donnees"
        os.makedirs(self.data_dir, exist_ok=True)
        for i in range(10):
            os.makedirs(f"{self.data_dir}/{i}", exist_ok=True)
        
        self.model = get_model("simple")
        try:
            self.model.load_state_dict(torch.load('personal_mnist_model.pth'))
            self.model.eval()
            print("✅ Personal model loaded")
        except:
            self.model = None
            print("❌ No personal model found")
        
        # Canvas setup
        self.pil_image = Image.new('L', (self.canvas_size, self.canvas_size), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        
        self.setup_ui()
    
    def setup_ui(self):
        # Title
        tk.Label(self.root, text="MNIST Feedback Learning", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Canvas
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, 
                               bg='black', cursor="pencil")
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # Main buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)
        
        tk.Button(btn_frame, text="Predict", command=self.predict, 
                 bg="blue", fg="white", width=10, font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Clear", command=self.clear, 
                 bg="red", fg="white", width=10, font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Results
        self.result_label = tk.Label(self.root, text="Draw a digit and click Predict", font=("Arial", 14))
        self.result_label.pack(pady=5)
        self.confidence_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.confidence_label.pack()
        
        # Feedback section
        feedback_frame = tk.LabelFrame(self.root, text="Feedback", font=("Arial", 12, "bold"))
        feedback_frame.pack(pady=15, padx=20, fill="x")
        
        fb_btn_frame = tk.Frame(feedback_frame)
        fb_btn_frame.pack(pady=10)
        
        self.correct_btn = tk.Button(fb_btn_frame, text="✅ Correct", command=self.mark_correct,
                                    bg="green", fg="white", width=12, state="disabled")
        self.correct_btn.pack(side=tk.LEFT, padx=5)
        
        self.wrong_btn = tk.Button(fb_btn_frame, text="❌ Wrong", command=self.mark_wrong,
                                  bg="orange", fg="white", width=12, state="disabled")
        self.wrong_btn.pack(side=tk.LEFT, padx=5)
        
        # Correction buttons
        tk.Label(feedback_frame, text="If wrong, click correct digit:").pack(pady=(10,5))
        correction_frame = tk.Frame(feedback_frame)
        correction_frame.pack()
        
        self.correction_buttons = []
        for i in range(10):
            btn = tk.Button(correction_frame, text=str(i), 
                           command=lambda d=i: self.correct_to(d),
                           width=3, height=1, state="disabled")
            btn.pack(side=tk.LEFT, padx=2)
            self.correction_buttons.append(btn)
        
        # Quick save
        save_frame = tk.LabelFrame(self.root, text="Quick Save")
        save_frame.pack(pady=10, padx=20, fill="x")
        
        save_btn_frame = tk.Frame(save_frame)
        save_btn_frame.pack(pady=5)
        for i in range(10):
            tk.Button(save_btn_frame, text=str(i), command=lambda d=i: self.quick_save(d),
                     width=3, height=1).pack(side=tk.LEFT, padx=2)
        
        # Stats
        stats_frame = tk.Frame(self.root)
        stats_frame.pack(pady=10)
        
        self.stats_label = tk.Label(stats_frame, text=f"Images: {self.saved_count}")
        self.stats_label.pack()
        self.corrections_label = tk.Label(stats_frame, text=f"Corrections: {self.corrections_count}", fg="blue")
        self.corrections_label.pack()
        
        self.retrain_btn = tk.Button(stats_frame, text="🔄 Re-train", command=self.retrain,
                                    bg="purple", fg="white", state="disabled")
        self.retrain_btn.pack(pady=5)
        
        # Status
        status_text = "Personal model loaded" if self.model else "No model loaded"
        color = "green" if self.model else "red"
        tk.Label(self.root, text=status_text, fg=color).pack(side=tk.BOTTOM, pady=5)
    
    def start_draw(self, e): self.drawing = True
    def stop_draw(self, e): self.drawing = False
    
    def draw(self, e):
        if self.drawing:
            x, y = e.x, e.y
            r = self.brush_size // 2
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
            self.pil_draw.ellipse([x-r, y-r, x+r, y+r], fill=255)
    
    def clear(self):
        self.canvas.delete("all")
        self.pil_image = Image.new('L', (self.canvas_size, self.canvas_size), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        self.result_label.config(text="Draw a digit and click Predict")
        self.confidence_label.config(text="")
        self.disable_feedback()
    
    def predict(self):
        if not self.model:
            self.result_label.config(text="No model loaded!", fg="red")
            return
        
        if np.array(self.pil_image).max() == 0:
            self.result_label.config(text="Draw something first!", fg="orange")
            return
        
        try:
            img = self.pil_image.resize((28, 28), Image.LANCZOS)
            tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)
            
            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(output, dim=1).item()
                conf = probs[0][pred].item() * 100
            
            self.current_prediction = pred
            self.current_confidence = conf
            
            self.result_label.config(text=f"Predicted: {pred}", fg="blue", font=("Arial", 16, "bold"))
            self.confidence_label.config(text=f"Confidence: {conf:.1f}%", 
                                        fg="green" if conf > 90 else "orange")
            self.enable_feedback()
            
        except Exception as e:
            self.result_label.config(text=f"Error: {str(e)}", fg="red")
    
    def enable_feedback(self):
        self.correct_btn.config(state="normal")
        self.wrong_btn.config(state="normal")
        for btn in self.correction_buttons:
            btn.config(state="normal")
    
    def disable_feedback(self):
        self.correct_btn.config(state="disabled")
        self.wrong_btn.config(state="disabled")
        for btn in self.correction_buttons:
            btn.config(state="disabled")
    
    def mark_correct(self):
        if self.current_prediction is not None:
            self.save_image(self.current_prediction)
            self.result_label.config(text=f"✅ Saved as {self.current_prediction}", fg="green")
            self.disable_feedback()
    
    def mark_wrong(self):
        self.result_label.config(text="❌ Click correct digit below:", fg="red")
        self.correct_btn.config(state="disabled")
        self.wrong_btn.config(state="disabled")
    
    def correct_to(self, digit):
        if self.current_prediction is not None:
            self.save_image(digit)
            self.corrections_count += 1
            self.corrections_label.config(text=f"Corrections: {self.corrections_count}")
            
            self.result_label.config(text=f"✅ Was {self.current_prediction}, saved as {digit}", fg="green")
            
            if self.corrections_count >= 1:
                self.retrain_btn.config(state="normal")
            
            self.disable_feedback()
    
    def quick_save(self, digit):
        if np.array(self.pil_image).max() == 0:
            messagebox.showwarning("Warning", "Draw something first!")
            return
        
        self.save_image(digit)
        self.result_label.config(text=f"Saved as {digit}!", fg="green")
    
    def save_image(self, digit):
        try:
            img = self.pil_image.resize((28, 28), Image.LANCZOS)
            filename = f"digit_{digit}_{int(time.time() * 1000)}.png"
            img.save(f"{self.data_dir}/{digit}/{filename}")
            
            self.saved_count += 1
            self.stats_label.config(text=f"Images: {self.saved_count}")
            print(f"Saved: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {e}")
    
    def retrain(self):
        if messagebox.askyesno("Re-train", f"Re-train with {self.corrections_count} corrections?"):
            self.result_label.config(text="🔄 Training...", fg="blue")
            self.retrain_btn.config(state="disabled", text="Training...")
            self.root.update()
            
            try:
                result = subprocess.run(['python', 'retrain.py'], capture_output=True)
                
                if result.returncode == 0:
                    # Reload model
                    try:
                        self.model.load_state_dict(torch.load('personal_mnist_model.pth'))
                        self.result_label.config(text="✅ Re-training successful!", fg="green")
                        self.corrections_count = 0
                        self.corrections_label.config(text="Corrections: 0")
                        self.retrain_btn.config(state="disabled")
                    except:
                        self.result_label.config(text="❌ Failed to reload model", fg="red")
                else:
                    self.result_label.config(text="❌ Training failed", fg="red")
                    
            except Exception as e:
                self.result_label.config(text=f"❌ Error: {e}", fg="red")
            
            self.retrain_btn.config(text="🔄 Re-train")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    print("MNIST Feedback Learning - Compact Version")
    print("Draw → Predict → Feedback → Re-train!")
    MNISTApp().run()