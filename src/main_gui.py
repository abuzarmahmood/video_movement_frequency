import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os
import threading
import json
from upload_to_s3 import delete_bucket_contents, validate_aws_credentials

class MainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Frequency Analysis")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Camera Process Sections
        self.setup_camera_frame(main_frame, 0, "Camera 1")
        self.setup_camera_frame(main_frame, 1, "Camera 2")
        
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
        
        # S3 Management Section
        s3_frame = ttk.LabelFrame(main_frame, text="S3 Management", padding="5")
        s3_frame.grid(row=4, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        self.purge_button = ttk.Button(s3_frame, text="Purge S3 Bucket", 
                                     command=self.purge_s3_bucket)
        self.purge_button.grid(row=0, column=0, pady=10)
        
        # Initialize state variables
        self.camera_states = {
            "cam1": {"running": False, "process": None},
            "cam2": {"running": False, "process": None}
        }
        self.freq_process = None

    def setup_camera_frame(self, parent, column, title):
        camera_frame = ttk.LabelFrame(parent, text=title, padding="5")
        camera_frame.grid(row=0, column=column, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Label(camera_frame, text="Camera Index:").grid(row=0, column=0, padx=5, pady=5)
        camera_index = ttk.Entry(camera_frame, width=10)
        camera_index.grid(row=0, column=1, padx=5, pady=5)
        camera_index.insert(0, str(column))
        
        ttk.Label(camera_frame, text="History Frames:").grid(row=1, column=0, padx=5, pady=5)
        n_history = ttk.Entry(camera_frame, width=10)
        n_history.grid(row=1, column=1, padx=5, pady=5)
        n_history.insert(0, "100")

        ttk.Label(camera_frame, text="Animal Number:").grid(row=2, column=0, padx=5, pady=5)
        animal_number = ttk.Entry(camera_frame, width=10)
        animal_number.grid(row=2, column=1, padx=5, pady=5)

        # Add radio buttons for camera options
        options_frame = ttk.LabelFrame(camera_frame, text="Options", padding="5")
        options_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        roi_var = tk.StringVar(value="none")
        ttk.Radiobutton(options_frame, text="No ROI", 
                       variable=roi_var, value="none").grid(row=0, column=0, padx=5)
        ttk.Radiobutton(options_frame, text="Use Saved ROI", 
                       variable=roi_var, value="saved").grid(row=0, column=1, padx=5)
        
        # Add CSV output options
        csv_var = tk.StringVar(value="append")
        ttk.Radiobutton(options_frame, text="Overwrite CSV", 
                       variable=csv_var, value="overwrite").grid(row=1, column=0, padx=5)
        ttk.Radiobutton(options_frame, text="Append CSV", 
                       variable=csv_var, value="append").grid(row=1, column=1, padx=5)
        
        camera_button = ttk.Button(
            camera_frame, 
            text="Start Camera", 
            command=lambda: self.toggle_camera(
                f"cam{column+1}",
                camera_index,
                n_history,
                animal_number,
                camera_button,
                roi_var,
                csv_var
            )
        )
        camera_button.grid(row=4, column=0, columnspan=2, pady=10)

    def toggle_camera(self, cam_id, camera_index, n_history, animal_number, button, roi_var, csv_var):
        if not self.camera_states[cam_id]["running"]:
            try:
                camera_idx = int(camera_index.get())
                n_hist = int(n_history.get())
                
                # Start camera process
                cmd = [sys.executable, 
                      os.path.join(os.path.dirname(__file__), "camera_process.py"),
                      str(camera_idx),
                      "--n_history", str(n_hist)]
                
                # Add animal number if provided
                animal_num = animal_number.get()
                if animal_num:
                    try:
                        animal_num = int(animal_num)
                        cmd.extend(["--animal-number", str(animal_num)])
                    except ValueError:
                        messagebox.showerror("Error", "Animal number must be a valid integer")
                        return

                # Add ROI options
                roi_option = roi_var.get()
                if roi_option == "saved":
                    cmd.append("--use-roi")
                    
                # Add CSV output options
                csv_option = csv_var.get()
                if csv_option == "append":
                    cmd.append("--append")
                elif csv_option == "overwrite":
                    cmd.append("--force-overwrite")
                
                self.camera_states[cam_id]["process"] = subprocess.Popen(cmd)
                self.camera_states[cam_id]["running"] = True
                button.configure(text="Stop Camera")
                self.status_label.configure(text=f"{cam_id} running")
                
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numeric values")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
        else:
            # Stop camera process
            if self.camera_states[cam_id]["process"]:
                self.camera_states[cam_id]["process"].terminate()
                self.camera_states[cam_id]["process"] = None
            self.camera_states[cam_id]["running"] = False
            button.configure(text="Start Camera")
            self.status_label.configure(text=f"{cam_id} stopped")

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

    def purge_s3_bucket(self):
        """Purge all contents from the S3 bucket after confirmation"""
        try:
            # Validate AWS credentials first
            validate_aws_credentials()
            
            # Ask for confirmation
            if messagebox.askyesno("Confirm Purge", 
                                 "Are you sure you want to purge all contents from the S3 bucket?"):
                # Get bucket name from environment or config
                bucket_name = os.getenv('AWS_S3_BUCKET')
                if not bucket_name:
                    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
                    if os.path.exists(config_path):
                        with open(config_path) as f:
                            config = json.load(f)
                            if 'aws' in config and 'AWS_S3_BUCKET' in config['aws']:
                                bucket_name = config['aws']['AWS_S3_BUCKET']
                
                if not bucket_name:
                    raise ValueError("AWS_S3_BUCKET not found in environment or config.json")
                
                # Perform the purge
                delete_bucket_contents(bucket_name)
                messagebox.showinfo("Success", "S3 bucket contents have been purged")
                self.status_label.configure(text="S3 bucket purged")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to purge S3 bucket: {str(e)}")

    def on_closing(self):
        # Cleanup processes
        for cam_id in self.camera_states:
            if self.camera_states[cam_id]["process"]:
                self.camera_states[cam_id]["process"].terminate()
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
