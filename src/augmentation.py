import os
import cv2
import numpy as np
import random
from collections import defaultdict, Counter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
from pathlib import Path
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

class YOLODataAugmentation:
    def __init__(self, dataset_path, classes_file, target_samples_per_class=500, output_path=None):
        """
        Initialize the YOLO data augmentation class
        
        Args:
            dataset_path (str): Path to the dataset containing images and labels
            classes_file (str): Path to classes.txt file
            target_samples_per_class (int): Target number of samples per class after augmentation
            output_path (str): Path where augmented dataset will be saved (default: /kaggle/working/)
        """
        self.dataset_path = Path(dataset_path)
        self.classes_file = classes_file
        self.target_samples_per_class = target_samples_per_class
        
        # Load classes
        self.classes = self._load_classes()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Initialize source paths (read-only)
        self.images_path = self.dataset_path / "images"
        self.labels_path = self.dataset_path / "labels"
        
        # Create augmented dataset paths in writable location
        if output_path is None:
            # Default to Kaggle working directory
            self.augmented_path = Path("/kaggle/working/augmented_dataset")
        else:
            self.augmented_path = Path(output_path)
            
        self.augmented_images_path = self.augmented_path / "images"
        self.augmented_labels_path = self.augmented_path / "labels"
        
        # Create directories
        self._create_directories()
        
        # Define augmentation pipeline
        self.augmentation_pipeline = self._create_augmentation_pipeline()
        
    def _load_classes(self):
        """Load classes from classes.txt file"""
        with open(self.classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes
    
    def _create_directories(self):
        """Create necessary directories for augmented dataset"""
        self.augmented_images_path.mkdir(parents=True, exist_ok=True)
        self.augmented_labels_path.mkdir(parents=True, exist_ok=True)
        print(f"Created output directories at: {self.augmented_path}")
    
    def _create_augmentation_pipeline(self):
        """Create comprehensive augmentation pipeline for industrial/CCTV scenarios"""
        return A.Compose([
            # Lighting and brightness variations (common in CCTV)
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.CLAHE(clip_limit=2.0, p=0.3),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            ], p=0.8),
            
            # Color variations
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            ], p=0.6),
            
            # Geometric transformations
            A.OneOf([
                A.Rotate(limit=15, p=0.5),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-10, 10), p=0.5),
                A.Perspective(scale=(0.02, 0.05), p=0.3),
            ], p=0.7),
            
            # Flipping (be careful with wire direction)
            A.HorizontalFlip(p=0.3),
            
            # Noise and blur (simulating camera quality issues)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.MotionBlur(blur_limit=(3, 5), p=0.3),
                A.MedianBlur(blur_limit=3, p=0.2),
            ], p=0.4),
            
            # Noise
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            ], p=0.3),
            
            # Occlusions and shadows
            A.OneOf([
                A.CoarseDropout(max_holes=3, max_height=32, max_width=32, p=0.3),
                A.GridDropout(ratio=0.2, unit_size_min=8, unit_size_max=16, p=0.2),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.3),
            ], p=0.4),
            
            # Weather effects (for outdoor scenarios)
            A.OneOf([
                A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, p=0.1),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
            ], p=0.1),
            
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def analyze_dataset_balance(self):
        """Analyze the current dataset balance"""
        class_counts = defaultdict(int)
        total_images = 0
        
        # Count objects per class across all images
        for label_file in self.labels_path.glob("*.txt"):
            total_images += 1
            if label_file.stat().st_size > 0:  # Non-empty file
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            class_name = self.classes[class_id]
                            class_counts[class_name] += 1
        
        print(f"Dataset Analysis:")
        print(f"Total images: {total_images}")
        print(f"Class distribution:")
        for class_name in self.classes:
            count = class_counts[class_name]
            print(f"  {class_name}: {count} objects")
        
        return dict(class_counts), total_images
    
    def _load_yolo_annotations(self, label_file):
        """Load YOLO format annotations"""
        annotations = []
        if label_file.exists() and label_file.stat().st_size > 0:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        annotations.append([x_center, y_center, width, height, class_id])
        return annotations
    
    def _save_yolo_annotations(self, annotations, label_file):
        """Save YOLO format annotations"""
        with open(label_file, 'w') as f:
            for ann in annotations:
                f.write(f"{ann[4]} {ann[0]:.6f} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f}\n")
    
    def _get_class_counts_per_image(self):
        """Get class counts for each image"""
        image_class_counts = {}
        
        for label_file in self.labels_path.glob("*.txt"):
            image_name = label_file.stem
            class_counts = defaultdict(int)
            
            if label_file.stat().st_size > 0:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
            
            image_class_counts[image_name] = dict(class_counts)
        
        return image_class_counts
    
    def augment_dataset(self):
        """Perform data augmentation to balance the dataset"""
        print("Starting data augmentation...")
        
        # Analyze current dataset
        class_counts, total_images = self.analyze_dataset_balance()
        image_class_counts = self._get_class_counts_per_image()
        
        # Calculate how many augmentations needed for each class
        max_count = max(class_counts.values()) if class_counts else 0
        target_count = max(max_count, self.target_samples_per_class)
        
        print(f"\nTarget samples per class: {target_count}")
        
        # First, copy original images from read-only input to writable output
        print("Copying original images...")
        for img_file in tqdm(self.images_path.glob("*")):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                shutil.copy2(img_file, self.augmented_images_path)
                
                # Copy corresponding label file
                label_file = self.labels_path / f"{img_file.stem}.txt"
                if label_file.exists():
                    shutil.copy2(label_file, self.augmented_labels_path)
                else:
                    # Create empty label file for images without annotations
                    (self.augmented_labels_path / f"{img_file.stem}.txt").touch()
        
        # Generate augmented images for each class
        augmentation_counter = 0
        
        for class_idx, class_name in enumerate(self.classes):
            current_count = class_counts.get(class_name, 0)
            needed_count = target_count - current_count
            
            if needed_count <= 0:
                print(f"Class '{class_name}' already has sufficient samples ({current_count})")
                continue
            
            print(f"\nGenerating {needed_count} augmented samples for class '{class_name}'...")
            
            # Find images that contain this class
            candidate_images = []
            for image_name, counts in image_class_counts.items():
                if class_idx in counts:
                    candidate_images.append(image_name)
            
            if not candidate_images:
                print(f"Warning: No images found containing class '{class_name}'")
                continue
            
            # Generate augmented images
            generated_count = 0
            max_attempts = needed_count * 3  # Avoid infinite loops
            attempts = 0
            
            while generated_count < needed_count and attempts < max_attempts:
                attempts += 1
                
                # Randomly select an image containing this class
                image_name = random.choice(candidate_images)
                img_file = self.images_path / f"{image_name}.jpg"
                
                # Try different extensions if .jpg doesn't exist
                if not img_file.exists():
                    for ext in ['.jpeg', '.png', '.bmp']:
                        img_file = self.images_path / f"{image_name}{ext}"
                        if img_file.exists():
                            break
                
                if not img_file.exists():
                    continue
                
                # Load image and annotations
                image = cv2.imread(str(img_file))
                if image is None:
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                label_file = self.labels_path / f"{image_name}.txt"
                annotations = self._load_yolo_annotations(label_file)
                
                if not annotations:
                    continue
                
                # Convert YOLO format to albumentations format
                bboxes = []
                class_labels = []
                
                for ann in annotations:
                    x_center, y_center, width, height, class_id = ann
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)
                
                # Apply augmentation
                try:
                    augmented = self.augmentation_pipeline(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    
                    aug_image = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_class_labels = augmented['class_labels']
                    
                    # Check if the target class is still present after augmentation
                    if class_idx not in aug_class_labels:
                        continue
                    
                    # Save augmented image
                    augmentation_counter += 1
                    aug_img_name = f"{image_name}_aug_{augmentation_counter}"
                    aug_img_file = self.augmented_images_path / f"{aug_img_name}.jpg"
                    
                    aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(aug_img_file), aug_image_bgr)
                    
                    # Save augmented annotations
                    aug_annotations = []
                    for bbox, cls_id in zip(aug_bboxes, aug_class_labels):
                        aug_annotations.append([bbox[0], bbox[1], bbox[2], bbox[3], cls_id])
                    
                    aug_label_file = self.augmented_labels_path / f"{aug_img_name}.txt"
                    self._save_yolo_annotations(aug_annotations, aug_label_file)
                    
                    generated_count += 1
                    
                except Exception as e:
                    print(f"Error during augmentation: {e}")
                    continue
            
            print(f"Generated {generated_count} augmented samples for class '{class_name}'")
        
        print(f"\nAugmentation complete! Generated {augmentation_counter} augmented images.")
        return self.augmented_path
    
    def split_dataset(self, train_ratio=0.8, val_ratio=0.2):
        """Split the augmented dataset into train and validation sets"""
        print("Splitting dataset into train/val...")
        
        # Create train/val directories
        train_images_path = self.augmented_path / "train" / "images"
        train_labels_path = self.augmented_path / "train" / "labels"
        val_images_path = self.augmented_path / "val" / "images"
        val_labels_path = self.augmented_path / "val" / "labels"
        
        for path in [train_images_path, train_labels_path, val_images_path, val_labels_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = list(self.augmented_images_path.glob("*"))
        image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        # Shuffle and split
        random.shuffle(image_files)
        train_size = int(len(image_files) * train_ratio)
        
        train_files = image_files[:train_size]
        val_files = image_files[train_size:]
        
        # Move files to respective directories
        for img_file in tqdm(train_files, desc="Moving train files"):
            shutil.move(str(img_file), str(train_images_path / img_file.name))
            
            label_file = self.augmented_labels_path / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.move(str(label_file), str(train_labels_path / label_file.name))
        
        for img_file in tqdm(val_files, desc="Moving val files"):
            shutil.move(str(img_file), str(val_images_path / img_file.name))
            
            label_file = self.augmented_labels_path / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.move(str(label_file), str(val_labels_path / label_file.name))
        
        print(f"Train set: {len(train_files)} images")
        print(f"Validation set: {len(val_files)} images")
        
        return train_images_path.parent, val_images_path.parent
    
    def create_data_yaml(self, train_path, val_path):
        """Create data.yaml file for YOLO training"""
        data_yaml = {
            'train': str(train_path),
            'val': str(val_path),
            'nc': len(self.classes),
            'names': self.classes
        }
        
        yaml_path = self.augmented_path / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"Created data.yaml at: {yaml_path}")
        return yaml_path
    
    def visualize_augmentations(self, num_samples=5):
        """Visualize some augmentation examples"""
        print("Generating visualization of augmentations...")
        
        # Find a sample image with annotations
        sample_images = []
        for label_file in self.labels_path.glob("*.txt"):
            if label_file.stat().st_size > 0:
                image_name = label_file.stem
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    img_file = self.images_path / f"{image_name}{ext}"
                    if img_file.exists():
                        sample_images.append((img_file, label_file))
                        break
        
        if not sample_images:
            print("No suitable images found for visualization")
            return
        
        # Select a random sample
        img_file, label_file = random.choice(sample_images)
        
        # Load image and annotations
        image = cv2.imread(str(img_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        annotations = self._load_yolo_annotations(label_file)
        
        # Convert to albumentations format
        bboxes = []
        class_labels = []
        
        for ann in annotations:
            x_center, y_center, width, height, class_id = ann
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(class_id)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Generate augmented versions
        for i in range(1, 6):
            try:
                augmented = self.augmentation_pipeline(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                
                axes[i].imshow(augmented['image'])
                axes[i].set_title(f"Augmented {i}")
                axes[i].axis('off')
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
                axes[i].set_title(f"Augmented {i} - Error")
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.augmented_path / "augmentation_examples.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def run_complete_pipeline(self):
        """Run the complete data augmentation pipeline"""
        print("=" * 50)
        print("YOLO Data Augmentation Pipeline")
        print("=" * 50)
        
        # Step 1: Analyze dataset
        class_counts, total_images = self.analyze_dataset_balance()
        
        # Step 2: Augment dataset
        augmented_path = self.augment_dataset()
        
        # Step 3: Split dataset
        train_path, val_path = self.split_dataset()
        
        # Step 4: Create data.yaml
        yaml_path = self.create_data_yaml(train_path, val_path)
        
        # Step 5: Visualize augmentations
        self.visualize_augmentations()
        
        print("\n" + "=" * 50)
        print("Pipeline Complete!")
        print("=" * 50)
        print(f"Augmented dataset: {augmented_path}")
        print(f"Train path: {train_path}")
        print(f"Val path: {val_path}")
        print(f"Data YAML: {yaml_path}")
        
        return {
            'augmented_path': augmented_path,
            'train_path': train_path,
            'val_path': val_path,
            'yaml_path': yaml_path
        }


# Example usage for Kaggle
if __name__ == "__main__":
    # Initialize the augmentation class with Kaggle paths
    augmentor = YOLODataAugmentation(
        dataset_path="/kaggle/input/peel-wire-yolo-dataset/yolo_dataset",  # Read-only input
        classes_file="/kaggle/input/peel-wire-yolo-dataset/yolo_dataset/classes.txt",
        target_samples_per_class=500,
        output_path="/kaggle/working/augmented_dataset"  # Writable output
    )
    
    # Run the complete pipeline
    results = augmentor.run_complete_pipeline()
    
    print("\nDataset is ready for YOLO training!")
    print(f"Use this command to train:")
    print(f"yolo train data={results['yaml_path']} model=yolov8n.pt epochs=100 imgsz=640")
    
    # Check disk usage
    import shutil
    total, used, free = shutil.disk_usage("/kaggle/working/")
    print(f"\nDisk usage:")
    print(f"Total: {total // (2**30)} GB")
    print(f"Used: {used // (2**30)} GB") 
    print(f"Free: {free // (2**30)} GB")