import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import gc
import psutil
import GPUtil
from IPython.display import clear_output
import cv2
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class YOLOWireDetectionTrainer:
    def __init__(self, input_dataset_path="/kaggle/input/peel-wire-augmented-dataset/kaggle/working/augmented_dataset", 
                 output_dir="/kaggle/working/yolo_training_output"):
        """
        Initialize YOLO training class for wire detection
        
        Args:
            input_dataset_path (str): Path to the input dataset (read-only)
            output_dir (str): Directory to save training outputs (working directory)
        """
        self.input_dataset_path = Path(input_dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a local data.yaml that points to the input dataset
        self.data_yaml_path = self.create_local_data_yaml()
        
        # Load data configuration
        with open(self.data_yaml_path, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        # Training parameters
        self.model_name = 'yolov8n.pt'  # Start with nano model
        self.img_size = 640
        self.batch_size = 16
        self.epochs = 100
        self.patience = 20
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Model variants for fallback
        self.model_variants = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
        
        # Classes
        self.classes = self.data_config['names']
        self.num_classes = self.data_config['nc']
        
        # Metrics storage
        self.training_history = defaultdict(list)
        
        print(f"Initialized trainer with:")
        print(f"  Device: {self.device}")
        print(f"  Classes: {self.classes}")
        print(f"  Model: {self.model_name}")
        print(f"  Image size: {self.img_size}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Input dataset: {self.input_dataset_path}")
        print(f"  Output directory: {self.output_dir}")
        
    def create_local_data_yaml(self):
        """
        Create a local data.yaml file that points to the input dataset paths
        This avoids copying the entire dataset to working directory
        """
        # Read the original data.yaml from input
        original_yaml_path = self.input_dataset_path / "data.yaml"
        
        if not original_yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found at {original_yaml_path}")
        
        with open(original_yaml_path, 'r') as f:
            original_config = yaml.safe_load(f)
        
        # Create new config with absolute paths pointing to input dataset
        new_config = {
            'train': str(self.input_dataset_path / "train"),
            'val': str(self.input_dataset_path / "val"),
            'nc': original_config['nc'],
            'names': original_config['names']
        }
        
        # Save the new data.yaml in working directory
        local_yaml_path = self.output_dir / "data.yaml"
        with open(local_yaml_path, 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False)
        
        print(f"Created local data.yaml at: {local_yaml_path}")
        print(f"  Train path: {new_config['train']}")
        print(f"  Val path: {new_config['val']}")
        
        # Verify paths exist
        train_path = Path(new_config['train'])
        val_path = Path(new_config['val'])
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training path not found: {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"Validation path not found: {val_path}")
        
        return local_yaml_path
        
    def check_system_resources(self):
        """Check available system resources"""
        print("\n=== System Resources ===")
        
        # CPU info
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"CPU Usage: {cpu_percent}%")
        
        # RAM info
        memory = psutil.virtual_memory()
        print(f"RAM: {memory.used/1e9:.2f}/{memory.total/1e9:.2f} GB ({memory.percent}%)")
        
        # GPU info
        if self.device == 'cuda':
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    print(f"GPU: {gpu.name}")
                    print(f"  Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB ({gpu.memoryUtil*100:.1f}%)")
                    print(f"  Temperature: {gpu.temperature}°C")
            except:
                print("GPU info not available")
        
        # Disk info
        working_disk = psutil.disk_usage('/kaggle/working')
        print(f"Working Disk: {working_disk.used/1e9:.2f}/{working_disk.total/1e9:.2f} GB ({working_disk.percent}%)")
        
        # Check dataset size (read-only)
        try:
            input_disk = psutil.disk_usage('/kaggle/input')
            print(f"Input Disk: {input_disk.used/1e9:.2f}/{input_disk.total/1e9:.2f} GB ({input_disk.percent}%)")
        except:
            print("Input disk info not available")
        
        print("=" * 30)

    def _parse_class_id(self, class_str):
        """
        Safely parse class ID from string, handling both int and float formats
        
        Args:
            class_str (str): Class ID as string (e.g., '0', '0.0', '1', '2.0')
            
        Returns:
            int: Parsed class ID
        """
        try:
            # First try to parse as float, then convert to int
            # This handles both '0' and '0.0' formats
            class_id = int(float(class_str))
            
            # Validate that class_id is within valid range
            if 0 <= class_id < len(self.classes):
                return class_id
            else:
                print(f"Warning: Invalid class ID {class_id}, skipping...")
                return None
        except (ValueError, TypeError):
            print(f"Warning: Could not parse class ID '{class_str}', skipping...")
            return None

    def check_dataset_integrity(self):
        """
        Check the integrity of the input dataset without modifying it
        Since input is read-only, we only report issues but don't fix them
        """
        print("Checking dataset integrity...")
        
        train_path = Path(self.data_config['train'])
        val_path = Path(self.data_config['val'])
        
        issues_found = 0
        total_files_checked = 0
        
        for dataset_name, dataset_path in [("Training", train_path), ("Validation", val_path)]:
            labels_path = dataset_path / 'labels'
            images_path = dataset_path / 'images'
            
            if not labels_path.exists() or not images_path.exists():
                print(f"⚠️  {dataset_name} path missing: {dataset_path}")
                continue
                
            for label_file in labels_path.glob('*.txt'):
                total_files_checked += 1
                
                if label_file.stat().st_size == 0:
                    continue
                
                # Check label file format
                with open(label_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                            
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = self._parse_class_id(parts[0])
                            if class_id is None:
                                print(f"⚠️  Issue in {label_file} line {line_num}: Invalid class ID")
                                issues_found += 1
                        else:
                            print(f"⚠️  Issue in {label_file} line {line_num}: Insufficient data")
                            issues_found += 1
        
        print(f"Dataset integrity check complete:")
        print(f"  Files checked: {total_files_checked}")
        print(f"  Issues found: {issues_found}")
        
        if issues_found > 0:
            print("⚠️  Note: Input dataset is read-only. Issues cannot be auto-fixed.")
            print("   Consider fixing the dataset source if training fails.")
        
    def analyze_dataset_distribution(self):
        """Analyze and visualize dataset distribution"""
        print("\nAnalyzing dataset distribution...")
        
        # Check dataset integrity (read-only check)
        self.check_dataset_integrity()
        
        # Count objects in train and val sets
        train_counts = defaultdict(int)
        val_counts = defaultdict(int)
        
        # Analyze training set
        train_labels_path = Path(self.data_config['train']) / 'labels'
        if train_labels_path.exists():
            for label_file in train_labels_path.glob('*.txt'):
                if label_file.stat().st_size > 0:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = self._parse_class_id(parts[0])
                                if class_id is not None:
                                    train_counts[self.classes[class_id]] += 1
        
        # Analyze validation set
        val_labels_path = Path(self.data_config['val']) / 'labels'
        if val_labels_path.exists():
            for label_file in val_labels_path.glob('*.txt'):
                if label_file.stat().st_size > 0:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = self._parse_class_id(parts[0])
                                if class_id is not None:
                                    val_counts[self.classes[class_id]] += 1
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Train distribution
        if train_counts:
            train_df = pd.DataFrame(list(train_counts.items()), columns=['Class', 'Count'])
            train_df = train_df.sort_values('Count', ascending=False)
            sns.barplot(data=train_df, x='Class', y='Count', ax=ax1)
            ax1.set_title('Training Set Distribution')
            ax1.set_xlabel('Class')
            ax1.set_ylabel('Number of Objects')
            ax1.tick_params(axis='x', rotation=45)
        else:
            ax1.text(0.5, 0.5, 'No training data found', ha='center', va='center')
            ax1.set_title('Training Set Distribution')
        
        # Val distribution
        if val_counts:
            val_df = pd.DataFrame(list(val_counts.items()), columns=['Class', 'Count'])
            val_df = val_df.sort_values('Count', ascending=False)
            sns.barplot(data=val_df, x='Class', y='Count', ax=ax2)
            ax2.set_title('Validation Set Distribution')
            ax2.set_xlabel('Class')
            ax2.set_ylabel('Number of Objects')
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No validation data found', ha='center', va='center')
            ax2.set_title('Validation Set Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dataset_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        train_images = len(list((Path(self.data_config['train']) / 'images').glob('*'))) if (Path(self.data_config['train']) / 'images').exists() else 0
        val_images = len(list((Path(self.data_config['val']) / 'images').glob('*'))) if (Path(self.data_config['val']) / 'images').exists() else 0
        
        print("\nDataset Statistics:")
        print(f"Training images: {train_images}")
        print(f"Validation images: {val_images}")
        print("\nClass distribution:")
        for class_name in self.classes:
            print(f"  {class_name}: Train={train_counts[class_name]}, Val={val_counts[class_name]}")
        
        # Check for potential issues
        if train_images == 0:
            print("\n⚠️  WARNING: No training images found!")
        if val_images == 0:
            print("\n⚠️  WARNING: No validation images found!")
        if sum(train_counts.values()) == 0:
            print("\n⚠️  WARNING: No training annotations found!")
        if sum(val_counts.values()) == 0:
            print("\n⚠️  WARNING: No validation annotations found!")
        
        # Save dataset analysis summary
        self.save_dataset_analysis(train_counts, val_counts, train_images, val_images)
        
    def save_dataset_analysis(self, train_counts, val_counts, train_images, val_images):
        """Save dataset analysis to file"""
        analysis_path = self.output_dir / 'dataset_analysis.txt'
        
        with open(analysis_path, 'w') as f:
            f.write("Dataset Analysis Summary\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input Dataset Path: {self.input_dataset_path}\n\n")
            
            f.write("Image Counts:\n")
            f.write(f"  Training images: {train_images}\n")
            f.write(f"  Validation images: {val_images}\n")
            f.write(f"  Total images: {train_images + val_images}\n\n")
            
            f.write("Object Counts by Class:\n")
            total_train_objects = sum(train_counts.values())
            total_val_objects = sum(val_counts.values())
            
            for class_name in self.classes:
                train_count = train_counts[class_name]
                val_count = val_counts[class_name]
                total_count = train_count + val_count
                f.write(f"  {class_name}:\n")
                f.write(f"    Train: {train_count}\n")
                f.write(f"    Val: {val_count}\n")
                f.write(f"    Total: {total_count}\n")
            
            f.write(f"\nTotal Objects:\n")
            f.write(f"  Training: {total_train_objects}\n")
            f.write(f"  Validation: {total_val_objects}\n")
            f.write(f"  Total: {total_train_objects + total_val_objects}\n")
        
        print(f"Dataset analysis saved to: {analysis_path}")
        
    def train_model(self, model_name=None, resume=False):
        """Train YOLO model with compatible parameters for current Ultralytics version"""
        if model_name:
            self.model_name = model_name
        
        print(f"\n{'Resuming' if resume else 'Starting'} training with {self.model_name}...")
        
        # Initialize model
        if resume and (self.output_dir / 'train' / 'weights' / 'last.pt').exists():
            model = YOLO(self.output_dir / 'train' / 'weights' / 'last.pt')
            print("Resumed from last checkpoint")
        else:
            model = YOLO(self.model_name)
        
        # Simplified training arguments compatible with current Ultralytics version
        train_args = {
            'data': str(self.data_yaml_path),
            'epochs': self.epochs,
            'imgsz': self.img_size,
            'batch': self.batch_size,
            'patience': self.patience,
            'save': True,
            'cache': False,  # Don't cache images to save memory
            'device': self.device,
            'workers': 2,  # Reduce workers for Kaggle
            'project': str(self.output_dir),
            'name': 'train',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
            'amp': True,  # Automatic mixed precision
            'fraction': 1.0,
            'profile': False,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': resume,
            'multi_scale': False,
            'fliplr': 0.5,
            'flipud': 0.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'auto_augment': 'randaugment',
            'erasing': 0.4,
        }
        
        # Clear memory before training
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        try:
            # Train the model
            results = model.train(**train_args)
            
            # Save final model
            self.model = model
            
            # Extract and save metrics
            self.extract_training_metrics()
            
            return True
            
        except Exception as e:
            print(f"Error during training: {e}")
            
            # Try with reduced batch size if OOM error
            if "out of memory" in str(e).lower() and self.batch_size > 4:
                print("Reducing batch size due to memory constraints...")
                self.batch_size = max(4, self.batch_size // 2)
                gc.collect()
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                return self.train_model(model_name, resume=False)
            
            return False
    
    def extract_training_metrics(self):
        """Extract and visualize training metrics"""
        print("\nExtracting training metrics...")
        
        # Read results.csv
        results_path = self.output_dir / 'train' / 'results.csv'
        if results_path.exists():
            try:
                df = pd.read_csv(results_path)
                df.columns = df.columns.str.strip()
                
                # Create comprehensive visualization
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                axes = axes.flatten()
                
                # Loss metrics
                loss_cols = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss']
                available_loss_cols = [col for col in loss_cols if col in df.columns]
                
                if available_loss_cols:
                    df[available_loss_cols].plot(ax=axes[0])
                    axes[0].set_title('Loss Metrics')
                    axes[0].set_xlabel('Epoch')
                    axes[0].set_ylabel('Loss')
                    axes[0].legend(loc='best')
                    axes[0].grid(True, alpha=0.3)
                else:
                    axes[0].text(0.5, 0.5, 'No loss metrics found', ha='center', va='center')
                    axes[0].set_title('Loss Metrics')
                
                # Precision and Recall
                metric_cols = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
                available_metric_cols = [col for col in metric_cols if col in df.columns]
                
                if available_metric_cols:
                    df[available_metric_cols].plot(ax=axes[1])
                    axes[1].set_title('Detection Metrics')
                    axes[1].set_xlabel('Epoch')
                    axes[1].set_ylabel('Score')
                    axes[1].legend(loc='best')
                    axes[1].grid(True, alpha=0.3)
                else:
                    axes[1].text(0.5, 0.5, 'No detection metrics found', ha='center', va='center')
                    axes[1].set_title('Detection Metrics')
                
                # Learning rate
                lr_cols = ['lr/pg0', 'lr/pg1', 'lr/pg2']
                available_lr_cols = [col for col in lr_cols if col in df.columns]
                
                if available_lr_cols:
                    df[available_lr_cols].plot(ax=axes[2])
                    axes[2].set_title('Learning Rate')
                    axes[2].set_xlabel('Epoch')
                    axes[2].set_ylabel('LR')
                    axes[2].legend(loc='best')
                    axes[2].grid(True, alpha=0.3)
                else:
                    axes[2].text(0.5, 0.5, 'No learning rate data found', ha='center', va='center')
                    axes[2].set_title('Learning Rate')
                
                # Box loss comparison
                if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
                    axes[3].plot(df.index, df['train/box_loss'], label='Train', alpha=0.8)
                    axes[3].plot(df.index, df['val/box_loss'], label='Val', alpha=0.8)
                    axes[3].set_title('Box Loss Comparison')
                    axes[3].set_xlabel('Epoch')
                    axes[3].set_ylabel('Loss')
                    axes[3].legend()
                    axes[3].grid(True, alpha=0.3)
                else:
                    axes[3].text(0.5, 0.5, 'Box loss data not available', ha='center', va='center')
                    axes[3].set_title('Box Loss Comparison')
                
                # Class loss comparison
                if 'train/cls_loss' in df.columns and 'val/cls_loss' in df.columns:
                    axes[4].plot(df.index, df['train/cls_loss'], label='Train', alpha=0.8)
                    axes[4].plot(df.index, df['val/cls_loss'], label='Val', alpha=0.8)
                    axes[4].set_title('Classification Loss Comparison')
                    axes[4].set_xlabel('Epoch')
                    axes[4].set_ylabel('Loss')
                    axes[4].legend()
                    axes[4].grid(True, alpha=0.3)
                else:
                    axes[4].text(0.5, 0.5, 'Classification loss data not available', ha='center', va='center')
                    axes[4].set_title('Classification Loss Comparison')
                
                # Final metrics bar plot
                if available_metric_cols:
                    final_metrics = df[available_metric_cols].iloc[-1]
                    final_metrics.plot(kind='bar', ax=axes[5])
                    axes[5].set_title('Final Metrics')
                    axes[5].set_ylabel('Score')
                    axes[5].set_xticklabels(axes[5].get_xticklabels(), rotation=45, ha='right')
                    axes[5].grid(True, alpha=0.3, axis='y')
                    
                    # Add value labels on bars
                    for i, (idx, val) in enumerate(final_metrics.items()):
                        axes[5].text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom')
                else:
                    axes[5].text(0.5, 0.5, 'No final metrics available', ha='center', va='center')
                    axes[5].set_title('Final Metrics')
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                # Print final metrics
                print("\n=== Final Training Metrics ===")
                if available_metric_cols:
                    for col in available_metric_cols:
                        print(f"{col}: {df[col].iloc[-1]:.4f}")
                
                # Save metrics to file
                self.save_training_summary(df)
                
            except Exception as e:
                print(f"Error reading training results: {e}")
        else:
            print("Training results file not found!")
    
    def save_training_summary(self, df):
        """Save comprehensive training summary"""
        summary_path = self.output_dir / 'training_summary.txt'
        
        try:
            with open(summary_path, 'w') as f:
                f.write("YOLO Wire Detection Training Summary\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {self.model_name}\n")
                f.write(f"Epochs: {self.epochs}\n")
                f.write(f"Batch Size: {self.batch_size}\n")
                f.write(f"Image Size: {self.img_size}\n")
                f.write(f"Device: {self.device}\n")
                f.write(f"Input Dataset: {self.input_dataset_path}\n\n")
                
                f.write("Classes:\n")
                for i, cls in enumerate(self.classes):
                    f.write(f"  {i}: {cls}\n")
                
                f.write("\nFinal Metrics:\n")
                metric_cols = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
                for col in metric_cols:
                    if col in df.columns:
                        f.write(f"  {col}: {df[col].iloc[-1]:.4f}\n")
                
                f.write("\nBest Metrics:\n")
                for col in metric_cols:
                    if col in df.columns:
                        best_val = df[col].max()
                        best_epoch = df[col].argmax()
                        f.write(f"  {col}: {best_val:.4f} (Epoch {best_epoch})\n")
            
            print(f"Training summary saved to: {summary_path}")
        except Exception as e:
            print(f"Error saving training summary: {e}")
    
    def validate_model(self, model_path=None):
        """Validate the trained model"""
        print("\nValidating model...")
        
        if model_path is None:
            model_path = self.output_dir / 'train' / 'weights' / 'best.pt'
        
        if not model_path.exists():
            print("Model weights not found!")
            return
        
        try:
            # Load model
            model = YOLO(model_path)
            
            # Validate
            metrics = model.val(
                data=str(self.data_yaml_path),
                imgsz=self.img_size,
                batch=self.batch_size,
                device=self.device,
                plots=True,
                save_json=True,
                conf=0.25,
                iou=0.45,
                max_det=300,
                half=True,
                split='val'
            )
            
            # Print validation results
            print("\n=== Validation Results ===")
            print(f"mAP50: {metrics.box.map50:.4f}")
            print(f"mAP50-95: {metrics.box.map:.4f}")
            print(f"Precision: {metrics.box.mp:.4f}")
            print(f"Recall: {metrics.box.mr:.4f}")
            
            # Per-class metrics
            print("\nPer-class mAP50:")
            for i, cls_map in enumerate(metrics.box.maps):
                print(f"  {self.classes[i]}: {cls_map:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"Error during validation: {e}")
            return None
    
    def inference_on_samples(self, num_samples=5):
        """Run inference on sample validation images"""
        print("\nRunning inference on sample images...")
        
        model_path = self.output_dir / 'train' / 'weights' / 'best.pt'
        if not model_path.exists():
            print("Model weights not found!")
            return
        
        try:
            # Load model
            model = YOLO(model_path)
            
            # Get sample validation images from input dataset
            val_images_path = Path(self.data_config['val']) / 'images'
            image_files = list(val_images_path.glob('*'))[:num_samples]
            
            if not image_files:
                print("No validation images found!")
                return
            
            # Create figure for visualization
            fig, axes = plt.subplots(1, min(num_samples, len(image_files)), figsize=(5*min(num_samples, len(image_files)), 5))
            if min(num_samples, len(image_files)) == 1:
                axes = [axes]
            
            for idx, img_file in enumerate(image_files):
                # Run inference
                results = model(img_file, conf=0.25, iou=0.45)
                
                # Plot results
                result_img = results[0].plot()
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                
                axes[idx].imshow(result_img)
                axes[idx].set_title(f'Sample {idx+1}')
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'inference_samples.png', dpi=150, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error during inference: {e}")
    
    def optimize_for_deployment(self):
        """Optimize model for deployment"""
        print("\nOptimizing model for deployment...")
        
        best_model_path = self.output_dir / 'train' / 'weights' / 'best.pt'
        if not best_model_path.exists():
            print("Best model weights not found!")
            return
        
        try:
            # Load model
            model = YOLO(best_model_path)
            
            # Export to different formats
            export_formats = {
                'onnx': {'imgsz': self.img_size, 'simplify': True, 'dynamic': False, 'opset': 12},
                'torchscript': {'imgsz': self.img_size, 'optimize': True},
            }
            
            for format_name, kwargs in export_formats.items():
                try:
                    print(f"\nExporting to {format_name}...")
                    model.export(format=format_name, **kwargs)
                    print(f"Successfully exported to {format_name}")
                except Exception as e:
                    print(f"Failed to export to {format_name}: {e}")
                    
        except Exception as e:
            print(f"Error during model optimization: {e}")
    
    def run_training_pipeline(self):
        """Run complete training pipeline"""
        print("=" * 60)
        print("YOLO Wire Detection Training Pipeline")
        print("=" * 60)
        print("🔄 Using space-efficient approach:")
        print("   • Reading dataset from input directory (no copying)")
        print("   • Saving only training outputs to working directory")
        print("=" * 60)
        
        # Step 1: Check system resources
        self.check_system_resources()
        
        # Step 2: Analyze dataset
        self.analyze_dataset_distribution()
        
        # Step 3: Train model
        success = self.train_model()
        
        if not success:
            print("\nTraining failed! Trying with smaller model...")
            # Try with different model variant
            for variant in self.model_variants[1:]:
                print(f"\nTrying with {variant}...")
                success = self.train_model(model_name=variant)
                if success:
                    break
        
        if success:
            # Step 4: Validate model
            self.validate_model()
            
            # Step 5: Run inference on samples
            self.inference_on_samples()
            
            # Step 6: Optimize for deployment
            self.optimize_for_deployment()
            
            # Step 7: Final resource check
            print("\nFinal resource usage:")
            self.check_system_resources()
            
            print("\n" + "=" * 60)
            print("Training Pipeline Complete!")
            print("=" * 60)
            print(f"💾 Model files saved to working directory:")
            print(f"   • Best model: {self.output_dir / 'train' / 'weights' / 'best.pt'}")
            print(f"   • Last model: {self.output_dir / 'train' / 'weights' / 'last.pt'}")
            print(f"   • Training metrics: {self.output_dir / 'training_metrics.png'}")
            print(f"   • Dataset analysis: {self.output_dir / 'dataset_analysis.txt'}")
            print(f"📁 Input dataset (read-only): {self.input_dataset_path}")
            
            # Provide deployment instructions
            self.print_deployment_instructions()
        
        else:
            print("\nTraining failed with all model variants!")
    
    def print_deployment_instructions(self):
        """Print instructions for model deployment"""
        print("\n" + "=" * 60)
        print("Model Deployment Instructions")
        print("=" * 60)
        
        print("\n1. For Python inference (Detection Only):")
        print("   from ultralytics import YOLO")
        print(f"   model = YOLO('{self.output_dir / 'train' / 'weights' / 'best.pt'}')")
        print("   results = model('path/to/image.jpg')")
        print("   # Returns bounding boxes for: intact_wire, machine, peeled_wire")
        
        print("\n2. For video processing (Detection Only):")
        print("   cap = cv2.VideoCapture('path/to/video.mp4')")
        print("   while True:")
        print("       ret, frame = cap.read()")
        print("       results = model(frame)")
        print("       # Extract detections: [x1, y1, x2, y2, confidence, class_id]")
        print("       # Classes: 0=intact_wire, 1=machine, 2=peeled_wire")
        
        print("\n3. IMPORTANT - Model Responsibility:")
        print("   ✓ YOLO Model: Detects objects (machine, intact_wire, peeled_wire)")
        print("   ✗ YOLO Model: Does NOT count wires")
        print("   → Wire counting is done by separate post-processing logic")
        
        print("\n4. Detection Output Format:")
        print("   for result in results:")
        print("       boxes = result.boxes  # Contains all detections")
        print("       for box in boxes:")
        print("           x1, y1, x2, y2 = box.xyxy[0]  # Bounding box")
        print("           conf = box.conf[0]  # Confidence score")
        print("           cls = int(box.cls[0])  # Class ID (0, 1, or 2)")
        
        print("\n5. Post-Processing Requirements:")
        print("   - Separate wire counting algorithm needed")
        print("   - Track spatial relationships between detections")
        print("   - Implement temporal tracking across frames")
        print("   - Count state transitions (intact → peeled)")
        
        print("\n6. Space-Efficient Training Setup:")
        print(f"   📂 Input Dataset: {self.input_dataset_path}")
        print(f"   💾 Model Outputs: {self.output_dir}")
        print("   ✅ No dataset duplication in working directory")


# Example usage with new directory structure
if __name__ == "__main__":
    # Initialize trainer with space-efficient setup
    trainer = YOLOWireDetectionTrainer(
        input_dataset_path="/kaggle/input/peel-wire-augmented-dataset/kaggle/working/augmented_dataset",
        output_dir="/kaggle/working/yolo_training_output"
    )
    
    # Run complete training pipeline
    trainer.run_training_pipeline()