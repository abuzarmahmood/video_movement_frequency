import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os
import threading
import json

class MainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Frequency Analysis")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Camera Process Section
        camera_frame = ttk.LabelFrame(main_frame, text="Camera Process", padding="5")
        camera_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Label(camera_frame, text="Camera Index:").grid(row=0, column=0, padx=5, pady=5)
        self.camera_index = ttk.Entry(camera_frame, width=10)
        self.camera_index.grid(row=0, column=1, padx=5, pady=5)
        self.camera_index.insert(0, "0")
        
        ttk.Label(camera_frame, text="History Frames:").grid(row=1, column=0, padx=5, pady=5)
        self.n_history = ttk.Entry(camera_frame, width=10)
        self.n_history.grid(row=1, column=1, padx=5, pady=5)
        self.n_history.insert(0, "100")

        ttk.Label(camera_frame, text="Animal Number:").grid(row=2, column=0, padx=5, pady=5)
        self.animal_number = ttk.Entry(camera_frame, width=10)
        self.animal_number.grid(row=2, column=1, padx=5, pady=5)
        
        self.camera_button = ttk.Button(camera_frame, text="Start Camera", command=self.toggle_camera)
        self.camera_button.grid(row=3, column=0, columnspan=2, pady=10)
        
        # ROI Selection Section
        roi_frame = ttk.LabelFrame(main_frame, text="ROI Selection", padding="5")
        roi_frame.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Label(roi_frame, text="Camera Index:").grid(row=0, column=0, padx=5, pady=5)
        self.roi_camera_index = ttk.Entry(roi_frame, width=10)
        self.roi_camera_index.grid(row=0, column=1, padx=5, pady=5)
        self.roi_camera_index.insert(0, "0")
        
        self.roi_button = ttk.Button(roi_frame, text="Select ROI", command=self.select_roi)
        self.roi_button.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Frequency Analysis Section
        freq_frame = ttk.LabelFrame(main_frame, text="Frequency Analysis", padding="5")
        freq_frame.grid(row=2, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        self.freq_button = ttk.Button(freq_frame, text="Start Frequency Analysis", 
                                    command=self.start_freq_analysis)
        self.freq_button.grid(row=0, column=0, pady=10)
        
        # Status section
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="5")
        status_frame.grid(row=3, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.grid(row=0, column=0, pady=5)
        
        # Initialize state variables
        self.camera_running = False
        self.camera_process = None
        self.freq_process = None

    def toggle_camera(self):
        if not self.camera_running:
            try:
                camera_idx = int(self.camera_index.get())
                n_history = int(self.n_history.get())
                
                # Start camera process
                cmd = [sys.executable, 
                      os.path.join(os.path.dirname(__file__), "camera_process.py"),
                      str(camera_idx),
                      "--n_history", str(n_history)]
                
                # Add animal number if provided
                animal_num = self.animal_number.get()
                if animal_num:
                    try:
                        animal_num = int(animal_num)
                        cmd.extend(["--animal-number", str(animal_num)])
                    except ValueError:
                        messagebox.showerror("Error", "Animal number must be a valid integer")
                        return
                
                self.camera_process = subprocess.Popen(cmd)
                self.camera_running = True
                self.camera_button.configure(text="Stop Camera")
                self.status_label.configure(text="Camera running")
                
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numeric values")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
        else:
            # Stop camera process
            if self.camera_process:
                self.camera_process.terminate()
                self.camera_process = None
            self.camera_running = False
            self.camera_button.configure(text="Start Camera")
            self.status_label.configure(text="Camera stopped")

    def start_freq_analysis(self):
        if self.freq_process and self.freq_process.poll() is None:
            messagebox.showinfo("Info", "Frequency analysis is already running")
            return
            
        try:
            # Start frequency analysis process
            cmd = [sys.executable, 
                  os.path.join(os.path.dirname(__file__), "process_freq.py")]
            
            self.freq_process = subprocess.Popen(cmd)
            self.status_label.configure(text="Frequency analysis running")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start frequency analysis: {str(e)}")

    def select_roi(self):
        try:
            camera_idx = int(self.roi_camera_index.get())
            
            # Start ROI selector process
            cmd = [sys.executable, 
                  os.path.join(os.path.dirname(__file__), "roi_selector.py"),
                  str(camera_idx)]
            
            process = subprocess.Popen(cmd)
            process.wait()  # Wait for ROI selection to complete
            
            # Update status
            self.status_label.configure(text="ROI selection completed")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid camera index")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start ROI selection: {str(e)}")

    def on_closing(self):
        # Cleanup processes
        if self.camera_process:
            self.camera_process.terminate()
        if self.freq_process:
            self.freq_process.terminate()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = MainGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
