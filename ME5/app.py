import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import torch
from ultralytics import YOLO
from PIL import Image, ImageTk
import numpy as np
from pathlib import Path
from datetime import datetime
import os
from queue import Queue
from threading import Thread, Event
import time

class VehicleDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ME 5: Vehicle Counter")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Set dark theme colors
        self.bg_color = '#1e1e1e'
        self.fg_color = '#ffffff'
        self.accent_color = '#2d2d2d'
        self.highlight_color = '#3d3d3d'
        
        # Configure styles
        self.setup_styles()
        
        # Initialize model
        self.model = YOLO('models/yolo11x.pt')
        
        # Variables
        self.setup_variables()
        
        # Setup UI
        self.setup_ui()
        
        # Setup video processing
        self.setup_video_processing()
        
    def setup_styles(self):
        style = ttk.Style()
        style.configure('TFrame', background=self.bg_color)
        style.configure('TLabel', background=self.bg_color, foreground=self.fg_color)
        style.configure('TLabelframe', background=self.bg_color)
        style.configure('TLabelframe.Label', background=self.bg_color, foreground=self.fg_color)
        style.configure('Custom.TButton', padding=5, background=self.accent_color)
        style.configure('Horizontal.TScale', background=self.bg_color, troughcolor=self.accent_color)
        self.root.configure(bg=self.bg_color)
        
    def setup_variables(self):
        self.current_image = None
        self.video_source = None
        self.is_video_playing = False
        self.video_paused = False
        self.confidence_threshold = tk.DoubleVar(value=0.25)
        self.iou_threshold = tk.DoubleVar(value=0.45)
        self.frame_skip = tk.IntVar(value=2)
        self.process_queue = Queue(maxsize=5)
        self.display_queue = Queue()
        self.stop_event = Event()
        self.username = "29yabuki"
        self.model_metrics = {
            'mAP': 0.0,
            'inference_speed': 0.0,
            'fps': 0.0
        }
        
    def setup_video_processing(self):
        # Start processing thread
        self.processing_thread = Thread(target=self.process_video_frames, daemon=True)
        self.processing_thread.start()
        
        # Start display thread
        self.display_thread = Thread(target=self.display_processed_frames, daemon=True)
        self.display_thread.start()
        
    def setup_ui(self):
        # Main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header with title and user info
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title on left
        title_label = ttk.Label(header_frame, text="ME 5: Vehicle Counter",
                            font=('Helvetica', 16, 'bold'))
        title_label.pack(side=tk.LEFT)
        
        # DateTime on right
        self.datetime_label = ttk.Label(header_frame, font=('Helvetica', 10))
        self.datetime_label.pack(side=tk.RIGHT)
        
        # Feed frame
        feed_frame = ttk.LabelFrame(main_container, text="Feed", padding="10")
        feed_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.display_label = ttk.Label(feed_frame, background=self.bg_color)
        self.display_label.pack(fill=tk.BOTH, expand=True)
        
        # Bottom section container
        bottom_frame = ttk.Frame(main_container)
        bottom_frame.pack(fill=tk.X)
        
        # Controls frame on left
        controls_frame = ttk.LabelFrame(bottom_frame, text="Controls", padding="10")
        controls_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # File controls
        file_frame = ttk.Frame(controls_frame)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Buttons container
        button_frame = ttk.Frame(file_frame)
        button_frame.pack(fill=tk.X)
        
        # Upload buttons on left, clear button on right
        self.image_btn = ttk.Button(button_frame, text="Upload Image", 
                                command=self.upload_image, style='Custom.TButton')
        self.image_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.video_btn = ttk.Button(button_frame, text="Upload Video", 
                                command=self.upload_video, style='Custom.TButton')
        self.video_btn.pack(side=tk.LEFT)
        
        self.clear_btn = ttk.Button(button_frame, text="Clear", 
                                command=self.clear_display, style='Custom.TButton')
        self.clear_btn.pack(side=tk.RIGHT)
        
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.pack(side=tk.RIGHT, padx=10)
        
        # Video controls
        self.create_video_controls(controls_frame)
        
        # Settings
        self.create_settings_frame(controls_frame)
        
        # Output frame on right
        output_frame = ttk.LabelFrame(bottom_frame, text="Output", padding="10", width=200)
        output_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))
        output_frame.pack_propagate(False)
        
        # Vehicle count
        self.count_label = ttk.Label(output_frame, text="Total Vehicles: 0",
                                font=('Helvetica', 14, 'bold'))
        self.count_label.pack(fill=tk.X)
        
        # Add model metrics display
        metrics_frame = ttk.Frame(output_frame)
        metrics_frame.pack(fill=tk.X, pady=(5, 0))
        
        # mAP display
        map_frame = ttk.Frame(metrics_frame)
        map_frame.pack(fill=tk.X, pady=2)
        ttk.Label(map_frame, text="mAP:").pack(side=tk.LEFT)
        self.map_label = ttk.Label(map_frame, text="0.0")
        self.map_label.pack(side=tk.RIGHT)
        
        # Speed display
        speed_frame = ttk.Frame(metrics_frame)
        speed_frame.pack(fill=tk.X, pady=2)
        ttk.Label(speed_frame, text="Inference:").pack(side=tk.LEFT)
        self.speed_label = ttk.Label(speed_frame, text="0.0")
        self.speed_label.pack(side=tk.RIGHT)
        
        # FPS display
        fps_frame = ttk.Frame(metrics_frame)
        fps_frame.pack(fill=tk.X, pady=2)
        ttk.Label(fps_frame, text="FPS:").pack(side=tk.LEFT)
        self.fps_label = ttk.Label(fps_frame, text="0.0")
        self.fps_label.pack(side=tk.RIGHT)
        
    def create_video_controls(self, parent):
        self.video_controls_frame = ttk.Frame(parent)
        self.video_controls_frame.pack(fill=tk.X, pady=(0, 10))
        self.video_controls_frame.pack_forget()
        
        # Video control buttons
        self.play_btn = ttk.Button(self.video_controls_frame, text="Play", 
                                 command=self.play_video, style='Custom.TButton')
        self.play_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.pause_btn = ttk.Button(self.video_controls_frame, text="Pause", 
                                  command=self.pause_video, style='Custom.TButton')
        self.pause_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_btn = ttk.Button(self.video_controls_frame, text="Stop", 
                                 command=self.stop_video, style='Custom.TButton')
        self.stop_btn.pack(side=tk.LEFT)
        
        # Frame skip control
        skip_frame = ttk.Frame(self.video_controls_frame)
        skip_frame.pack(side=tk.LEFT, padx=(20, 0))
        
        ttk.Label(skip_frame, text="Process every nth frame:").pack(side=tk.LEFT)
        frame_skip_spin = ttk.Spinbox(skip_frame, from_=1, to=10, 
                                    width=5, textvariable=self.frame_skip)
        frame_skip_spin.pack(side=tk.LEFT, padx=5)
        
    def create_settings_frame(self, parent):
        settings_frame = ttk.LabelFrame(parent, text="Detection Settings", padding="5")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Confidence threshold
        conf_frame = ttk.Frame(settings_frame)
        conf_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(conf_frame, text="Confidence:").pack(side=tk.LEFT)
        self.conf_scale = ttk.Scale(conf_frame, from_=0.0, to=1.0, 
                                  orient=tk.HORIZONTAL, variable=self.confidence_threshold)
        self.conf_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.conf_label = ttk.Label(conf_frame, text="0.25", width=5)
        self.conf_label.pack(side=tk.LEFT)
        
        # IoU threshold
        iou_frame = ttk.Frame(settings_frame)
        iou_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(iou_frame, text="IoU:").pack(side=tk.LEFT)
        self.iou_scale = ttk.Scale(iou_frame, from_=0.0, to=1.0, 
                                 orient=tk.HORIZONTAL, variable=self.iou_threshold)
        self.iou_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.iou_label = ttk.Label(iou_frame, text="0.45", width=5)
        self.iou_label.pack(side=tk.LEFT)
        
        # Bind threshold changes
        self.conf_scale.configure(command=self.on_threshold_change)
        self.iou_scale.configure(command=self.on_threshold_change)
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        
        if file_path:
            self.stop_video()
            self.video_controls_frame.pack_forget()
            self.current_image = file_path
            self.file_label.config(text=f"Image: {Path(file_path).name}")
            self.process_image(file_path)
            
    def upload_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi")])
        
        if file_path:
            self.stop_video()
            if self.video_source:
                self.video_source.release()
            
            self.video_source = cv2.VideoCapture(file_path)
            self.file_label.config(text=f"Video: {Path(file_path).name}")
            self.video_controls_frame.pack()
            self.current_image = None
            
    def process_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Failed to load image")
            
            # Time the inference
            start_time = time.time()
            results = self.model(img, conf=self.confidence_threshold.get(), 
                               iou=self.iou_threshold.get())[0]
            end_time = time.time()
            
            # Calculate metrics
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            fps = 1 / (end_time - start_time)
            
            # Update metrics
            self.model_metrics['inference_speed'] = inference_time
            self.model_metrics['fps'] = fps
            
            # Update labels
            self.speed_label.config(text=f"{inference_time:.1f}")
            self.fps_label.config(text=f"{fps:.1f}")
            
            annotated_img = self.draw_detections(img, results)
            self.display_image(annotated_img)
            
        except Exception as e:
            print(f"Error processing image: {e}")
            
    def process_video_frames(self):
        while not self.stop_event.is_set():
            if not self.is_video_playing or self.video_paused:
                time.sleep(0.1)
                continue
                
            try:
                frame = self.process_queue.get()
                if frame is None:
                    continue
                
                # Time the inference
                start_time = time.time()
                results = self.model(frame, conf=self.confidence_threshold.get(), 
                                  iou=self.iou_threshold.get())[0]
                end_time = time.time()
                
                # Calculate metrics
                inference_time = (end_time - start_time) * 1000  # Convert to ms
                fps = 1 / (end_time - start_time)
                
                # Update metrics
                self.model_metrics['inference_speed'] = inference_time
                self.model_metrics['fps'] = fps
                
                # Update labels
                self.speed_label.config(text=f"{inference_time:.1f}")
                self.fps_label.config(text=f"{fps:.1f}")
                
                annotated_frame = self.draw_detections(frame, results)
                self.display_queue.put(annotated_frame)
                
            except Exception as e:
                print(f"Error processing video frame: {e}")
                
    def display_processed_frames(self):
        while not self.stop_event.is_set():
            try:
                if not self.display_queue.empty():
                    frame = self.display_queue.get()
                    if frame is not None:
                        self.display_image(frame)
                time.sleep(0.01)
            except Exception as e:
                print(f"Error displaying frame: {e}")
                
    def draw_detections(self, image, results):
        annotated_img = image.copy()
        vehicle_count = 0
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            
            if cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                vehicle_count += 1
                
                # Draw box with thickness based on confidence
                thickness = max(1, int(conf * 3))
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), thickness)
                
                # Add label with confidence
                label = f'{results.names[cls]} {conf:.2f}'
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                cv2.rectangle(annotated_img, (x1, y1 - label_height - 5),
                            (x1 + label_width, y1), (0, 0, 255), -1)
                cv2.putText(annotated_img, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Update count label
        self.count_label.config(text=f"Total Vehicles: {vehicle_count}")
        
        return annotated_img
        
    def play_video(self):
        if self.video_source and not self.is_video_playing:
            self.is_video_playing = True
            self.video_paused = False
            self.update_video_frame()
            
    def pause_video(self):
        self.video_paused = not self.video_paused
        
    def stop_video(self):
        self.is_video_playing = False
        self.video_paused = False
        if self.video_source:
            self.video_source.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def update_video_frame(self):
        if not self.is_video_playing or not self.video_source:
            return
            
        if self.video_paused:
            self.root.after(30, self.update_video_frame)
            return
            
        ret, frame = self.video_source.read()
        if ret:
            # Skip frames based on frame_skip value
            for _ in range(self.frame_skip.get() - 1):
                self.video_source.read()
                
            if not self.process_queue.full():
                self.process_queue.put(frame.copy())
                
            self.root.after(30, self.update_video_frame)
        else:
            self.video_source.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.is_video_playing = False
            
    def clear_display(self):
        self.stop_video()
        if self.video_source:
            self.video_source.release()
            self.video_source = None
        
        self.current_image = None
        self.file_label.config(text="No file selected")
        self.video_controls_frame.pack_forget()
        self.display_label.configure(image='')
        self.display_label.image = None
        self.count_label.config(text="Total Vehicles: 0")
            
    def display_image(self, cv2_image):
        try:
            rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            
            # Set fixed display size or maintain original aspect ratio
            desired_width = 800  # Fixed width
            desired_height = 450  # Fixed height
            
            # Calculate scaling while maintaining aspect ratio
            height, width = rgb_image.shape[:2]
            scale = min(desired_width/width, desired_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image
            resized_image = cv2.resize(rgb_image, (new_width, new_height),
                                    interpolation=cv2.INTER_AREA)
            
            # Convert to PhotoImage
            image = Image.fromarray(resized_image)
            photo = ImageTk.PhotoImage(image)
            
            # Update display
            self.display_label.configure(image=photo)
            self.display_label.image = photo
            
        except Exception as e:
            print(f"Error displaying image: {e}")
            
    def on_threshold_change(self, *args):
        self.conf_label.config(text=f"{self.confidence_threshold.get():.2f}")
        self.iou_label.config(text=f"{self.iou_threshold.get():.2f}")
        
        if self.current_image and not self.is_video_playing:
            self.process_image(self.current_image)
            
    def on_closing(self):
        self.stop_event.set()
        if self.video_source:
            self.video_source.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VehicleDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()