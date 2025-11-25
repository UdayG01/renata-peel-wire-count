import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time
from datetime import datetime
import argparse
import os

class YOLOVideoInference:
    def __init__(self, model_path, confidence=0.25, iou_threshold=0.45, device='auto'):
        """
        Initialize YOLO video inference
        
        Args:
            model_path (str): Path to trained YOLO model (.pt file)
            confidence (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
            device (str): Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self.load_model()
        
        # Class names and colors
        self.class_names = ['intact_wire', 'machine', 'peeled_wire']
        self.colors = {
            0: (0, 255, 0),    # Green for intact_wire
            1: (255, 0, 0),    # Blue for machine
            2: (0, 0, 255)     # Red for peeled_wire
        }
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'total_detections': 0,
            'class_counts': {name: 0 for name in self.class_names},
            'processing_time': 0,
            'fps': 0
        }
        
    def load_model(self):
        """Load the trained YOLO model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading model from: {self.model_path}")
        model = YOLO(self.model_path)
        
        # Move model to device
        if self.device == 'cuda' and torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        return model
    
    def draw_detections(self, frame, results):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            results: YOLO detection results
            
        Returns:
            annotated_frame: Frame with drawn detections
        """
        annotated_frame = frame.copy()
        
        if results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Get box coordinates
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                # Skip if confidence below threshold
                if confidence < self.confidence:
                    continue
                
                # Get class name and color
                class_name = self.class_names[class_id]
                color = self.colors[class_id]
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label
                label = f"{class_name}: {confidence:.2f}"
                
                # Calculate label size and position
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Draw label background
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - label_height - 10),
                    (x1 + label_width, y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                
                # Update statistics
                self.stats['total_detections'] += 1
                self.stats['class_counts'][class_name] += 1
        
        return annotated_frame
    
    def add_info_overlay(self, frame, frame_num, fps, total_frames):
        """Add information overlay to frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Info box background
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text information
        info_lines = [
            f"Frame: {frame_num}/{total_frames}",
            f"FPS: {fps:.1f}",
            f"Detections: {self.stats['total_detections']}",
            f"Device: {self.device.upper()}"
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 20
            cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add class counts
        y_start = 130
        cv2.putText(frame, "Detections:", (20, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for i, (class_name, count) in enumerate(self.stats['class_counts'].items()):
            color = self.colors[i]
            y_pos = y_start + 20 + i * 15
            cv2.putText(frame, f"{class_name}: {count}", (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def process_video(self, input_video_path, output_video_path=None, show_preview=False, save_stats=True):
        """
        Process video with YOLO inference
        
        Args:
            input_video_path (str): Path to input video
            output_video_path (str): Path to output video (optional)
            show_preview (bool): Show real-time preview
            save_stats (bool): Save processing statistics
        """
        input_path = Path(input_video_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        # Open input video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Input video: {input_path}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")
        
        # Setup output video if specified
        out = None
        if output_video_path:
            output_path = Path(output_video_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"Output video: {output_path}")
        
        # Reset statistics
        self.stats = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'total_detections': 0,
            'class_counts': {name: 0 for name in self.class_names},
            'processing_time': 0,
            'fps': 0
        }
        
        # Processing loop
        frame_count = 0
        start_time = time.time()
        last_fps_time = start_time
        fps_frame_count = 0
        running_fps = 0
        
        print("\nProcessing video...")
        print("Press 'q' to quit preview, 'p' to pause")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Reached end of video or failed to read frame")
                    break
                
                frame_count += 1
                fps_frame_count += 1
                frame_start_time = time.time()
                
                # Run YOLO inference
                try:
                    results = self.model(
                        frame,
                        conf=self.confidence,
                        iou=self.iou_threshold,
                        device=self.device,
                        verbose=False
                    )
                except Exception as e:
                    print(f"Error during inference on frame {frame_count}: {e}")
                    continue
                
                # Draw detections
                try:
                    annotated_frame = self.draw_detections(frame, results)
                except Exception as e:
                    print(f"Error drawing detections on frame {frame_count}: {e}")
                    annotated_frame = frame
                
                # Calculate FPS more robustly
                frame_time = time.time() - frame_start_time
                
                # Update running FPS every 10 frames
                if fps_frame_count >= 10:
                    elapsed = time.time() - last_fps_time
                    if elapsed > 0:
                        running_fps = fps_frame_count / elapsed
                    fps_frame_count = 0
                    last_fps_time = time.time()
                
                # Add info overlay
                display_fps = running_fps if running_fps > 0 else (1.0 / frame_time if frame_time > 0 else 0)
                try:
                    annotated_frame = self.add_info_overlay(annotated_frame, frame_count, display_fps, total_frames)
                except Exception as e:
                    print(f"Error adding overlay to frame {frame_count}: {e}")
                
                # Save frame if output video specified
                if out:
                    try:
                        out.write(annotated_frame)
                    except Exception as e:
                        print(f"Error writing frame {frame_count} to output: {e}")
                        break
                
                # Show preview if requested
                if show_preview:
                    try:
                        # Resize for display if too large
                        display_frame = annotated_frame
                        if width > 1280:
                            scale = 1280 / width
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            display_frame = cv2.resize(annotated_frame, (new_width, new_height))
                        
                        cv2.imshow('YOLO Inference', display_frame)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            print("\nStopping video processing...")
                            break
                        elif key == ord('p'):
                            print("Paused. Press any key to continue...")
                            cv2.waitKey(0)
                    except Exception as e:
                        print(f"Error displaying frame {frame_count}: {e}")
                        show_preview = False  # Disable preview on error
                
                # Update statistics
                self.stats['processed_frames'] = frame_count
                
                # Print progress more frequently for shorter videos
                progress_interval = min(30, max(1, total_frames // 20))  # At least every 5% or every 30 frames
                
                if frame_count % progress_interval == 0 or frame_count == total_frames:
                    progress = (frame_count / total_frames) * 100
                    elapsed_time = time.time() - start_time
                    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                    eta_seconds = (total_frames - frame_count) / avg_fps if avg_fps > 0 else 0
                    eta_minutes = eta_seconds / 60
                    
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) | "
                          f"Avg FPS: {avg_fps:.1f} | "
                          f"Detections: {self.stats['total_detections']} | "
                          f"ETA: {eta_minutes:.1f}m")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            if show_preview:
                cv2.destroyAllWindows()
            
            # Final statistics
            total_time = time.time() - start_time
            self.stats['processing_time'] = total_time
            self.stats['fps'] = frame_count / total_time if total_time > 0 else 0
            
            # Print final statistics
            self.print_statistics()
            
            # Save statistics if requested
            if save_stats and output_video_path:
                self.save_statistics(output_video_path)
    
    def print_statistics(self):
        """Print processing statistics"""
        print("\n" + "=" * 50)
        print("PROCESSING STATISTICS")
        print("=" * 50)
        print(f"Processed frames: {self.stats['processed_frames']}/{self.stats['total_frames']}")
        print(f"Processing time: {self.stats['processing_time']:.2f} seconds")
        print(f"Average FPS: {self.stats['fps']:.2f}")
        print(f"Total detections: {self.stats['total_detections']}")
        print("\nDetections by class:")
        for class_name, count in self.stats['class_counts'].items():
            percentage = (count / self.stats['total_detections'] * 100) if self.stats['total_detections'] > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        print("=" * 50)
    
    def save_statistics(self, output_video_path):
        """Save statistics to file"""
        output_path = Path(output_video_path)
        stats_path = output_path.parent / f"{output_path.stem}_stats.txt"
        
        with open(stats_path, 'w') as f:
            f.write("YOLO Video Inference Statistics\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Input video: {output_path.parent / 'input_video'}\n")
            f.write(f"Output video: {output_path}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Confidence threshold: {self.confidence}\n")
            f.write(f"IoU threshold: {self.iou_threshold}\n\n")
            
            f.write("Processing Results:\n")
            f.write(f"  Processed frames: {self.stats['processed_frames']}/{self.stats['total_frames']}\n")
            f.write(f"  Processing time: {self.stats['processing_time']:.2f} seconds\n")
            f.write(f"  Average FPS: {self.stats['fps']:.2f}\n")
            f.write(f"  Total detections: {self.stats['total_detections']}\n\n")
            
            f.write("Detections by class:\n")
            for class_name, count in self.stats['class_counts'].items():
                percentage = (count / self.stats['total_detections'] * 100) if self.stats['total_detections'] > 0 else 0
                f.write(f"  {class_name}: {count} ({percentage:.1f}%)\n")
        
        print(f"Statistics saved to: {stats_path}")


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='YOLO Video Inference')
    parser.add_argument('--model', required=True, help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--input', required=True, help='Path to input video')
    parser.add_argument('--output', help='Path to output video (optional)')
    parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'], help='Device to use')
    parser.add_argument('--preview', action='store_true', help='Show real-time preview')
    parser.add_argument('--no-stats', action='store_true', help='Don\'t save statistics')
    
    args = parser.parse_args()
    
    # Initialize inference
    inferencer = YOLOVideoInference(
        model_path=args.model,
        confidence=args.confidence,
        iou_threshold=args.iou,
        device=args.device
    )
    
    # Process video
    inferencer.process_video(
        input_video_path=args.input,
        output_video_path=args.output,
        show_preview=args.preview,
        save_stats=not args.no_stats
    )


# Example usage for Kaggle/Jupyter
if __name__ == "__main__":
    # For interactive use (Jupyter/Kaggle)
    if (1 == 2):
        # Jupyter/Kaggle environment
        
        # Example configuration
        MODEL_PATH = "/kaggle/working/yolo_training_output/train/weights/best.pt"
        INPUT_VIDEO = "/kaggle/input/your-video-dataset/test_video.mp4"  # Update this path
        OUTPUT_VIDEO = "/kaggle/working/output_with_detections.mp4"
        
        # Initialize inference
        inferencer = YOLOVideoInference(
            model_path=MODEL_PATH,
            confidence=0.25,
            iou_threshold=0.45,
            device='auto'  # Will use GPU if available
        )
        
        # Process video
        print("Starting video inference...")
        inferencer.process_video(
            input_video_path=INPUT_VIDEO,
            output_video_path=OUTPUT_VIDEO,
            show_preview=False,  # Set to True if you want to see preview
            save_stats=True
        )
        
        print(f"\nProcessing complete!")
        print(f"Output video saved to: {OUTPUT_VIDEO}")
        
    else:
        # Command line usage
        main()