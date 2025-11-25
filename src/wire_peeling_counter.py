import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time
from datetime import datetime
import argparse
from collections import deque, defaultdict
import json

class WirePeelingCounter:
    def __init__(self, model_path, confidence=0.25, iou_threshold=0.45, device='auto'):
        """
        Initialize Wire Peeling Counter with post-processing pipeline
        
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
        
        # Class mapping
        self.class_names = {0: 'intact_wire', 1: 'machine', 2: 'peeled_wire'}
        self.class_ids = {'intact_wire': 0, 'machine': 1, 'peeled_wire': 2}
        
        # Tracking parameters
        self.machine_area = None
        self.machine_area_buffer = 125  # pixels to expand machine bounding box
        self.machine_detection_frames = deque(maxlen=10)  # Store recent machine detections
        
        # Wire tracking
        self.wire_states = {}  # Track individual wires
        self.next_wire_id = 0
        self.max_wire_distance = 150  # Maximum distance for wire tracking between frames
        
        # Counting logic
        self.peeling_count = 0
        self.peeling_operations = []  # Store details of each operation
        self.current_peeling_state = False
        self.peeling_cooldown = 0
        self.peeling_cooldown_frames = 5   
        self.wire_cooldown = {}  # Per-wire cooldown to prevent same wire being counted multiple times
        
        # Visual feedback
        self.monitoring_box_color = (0, 255, 0)  # Green for monitoring
        self.detection_box_color = (0, 0, 255)   # Red for detection
        self.box_flash_duration = 15  # Frames to show red box
        self.box_flash_counter = 0
        
        # Statistics
        self.frame_count = 0
        self.total_detections = {'intact_wire': 0, 'machine': 0, 'peeled_wire': 0}
        self.operations_log = []
        
    def load_model(self):
        """Load the trained YOLO model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading model from: {self.model_path}")
        model = YOLO(self.model_path)
        return model
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def point_in_box(self, point, box):
        """Check if a point is inside a bounding box"""
        x, y = point
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def expand_box(self, box, buffer):
        """Expand bounding box by buffer pixels"""
        x1, y1, x2, y2 = box
        return (max(0, x1 - buffer), max(0, y1 - buffer), x2 + buffer, y2 + buffer)
    
    def update_machine_area(self, detections):
        """Update the machine monitoring area based on recent detections"""
        machine_boxes = []
        
        for detection in detections:
            if detection['class'] == 'machine' and detection['confidence'] > self.confidence:
                machine_boxes.append(detection['bbox'])
        
        if machine_boxes:
            # Add to recent detections
            self.machine_detection_frames.append(machine_boxes)
            
            # Calculate average machine area from recent frames
            all_boxes = []
            for frame_boxes in self.machine_detection_frames:
                all_boxes.extend(frame_boxes)
            
            if all_boxes:
                # Calculate union of all machine boxes
                x1_min = min(box[0] for box in all_boxes)
                y1_min = min(box[1] for box in all_boxes)
                x2_max = max(box[2] for box in all_boxes)
                y2_max = max(box[3] for box in all_boxes)
                
                # Expand the area with buffer
                self.machine_area = self.expand_box(
                    (x1_min, y1_min, x2_max, y2_max), 
                    self.machine_area_buffer
                )
    
    def track_wires(self, detections):
        """Track individual wires across frames"""
        current_wires = {'intact_wire': [], 'peeled_wire': []}
        
        # Extract wire detections
        for detection in detections:
            if detection['class'] in ['intact_wire', 'peeled_wire']:
                current_wires[detection['class']].append(detection)
        
        # Update wire tracking
        for wire_type in ['intact_wire', 'peeled_wire']:
            for detection in current_wires[wire_type]:
                self.match_wire_to_existing(detection, wire_type)
        
        # Clean up old wires
        self.cleanup_old_wires()
    
    def match_wire_to_existing(self, detection, wire_type):
        """Match current detection to existing wire or create new one"""
        bbox = detection['bbox']
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        best_match_id = None
        min_distance = float('inf')
        
        # Find closest existing wire
        for wire_id, wire_data in self.wire_states.items():
            if wire_data['last_seen_frame'] >= self.frame_count - 5:  # Recently seen
                last_center = wire_data['last_center']
                distance = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
                
                if distance < self.max_wire_distance and distance < min_distance:
                    min_distance = distance
                    best_match_id = wire_id
        
        if best_match_id is not None:
            # Update existing wire
            self.wire_states[best_match_id].update({
                'last_center': center,
                'last_bbox': bbox,
                'last_seen_frame': self.frame_count,
                'current_type': wire_type,
                'confidence': detection['confidence']
            })
            
            # Check for state transition
            if (self.wire_states[best_match_id].get('previous_type') == 'intact_wire' and 
                wire_type == 'peeled_wire'):
                self.check_peeling_operation(best_match_id)
            
            self.wire_states[best_match_id]['previous_type'] = wire_type
        else:
            # Create new wire
            self.wire_states[self.next_wire_id] = {
                'id': self.next_wire_id,
                'first_seen_frame': self.frame_count,
                'last_seen_frame': self.frame_count,
                'first_center': center,
                'last_center': center,
                'last_bbox': bbox,
                'current_type': wire_type,
                'previous_type': None,
                'confidence': detection['confidence'],
                'in_machine_area': self.machine_area and self.point_in_box(center, self.machine_area)
            }
            self.next_wire_id += 1
    
    def check_peeling_operation(self, wire_id):
        """Check if a wire transition represents a successful peeling operation"""
        # Check per-wire cooldown first
        if wire_id in self.wire_cooldown and self.wire_cooldown[wire_id] > 0:
            return False
        
        # Check global cooldown (reduced impact)
        if self.peeling_cooldown > 0:
            return False
        
        wire = self.wire_states[wire_id]
        
        # Check if wire was in machine area during transition
        if (wire.get('in_machine_area', False) and 
            self.machine_area and 
            self.point_in_box(wire['last_center'], self.machine_area)):
            
            # Valid peeling operation detected
            self.peeling_count += 1
            self.current_peeling_state = True
            self.box_flash_counter = self.box_flash_duration
            self.peeling_cooldown = self.peeling_cooldown_frames
            
            # Set per-wire cooldown (prevents same wire from being counted again)
            self.wire_cooldown[wire_id] = 20  # Longer cooldown for same wire
            
            # Log the operation
            operation = {
                'frame': self.frame_count,
                'wire_id': wire_id,
                'timestamp': time.time(),
                'machine_area': self.machine_area,
                'wire_position': wire['last_center']
            }
            self.operations_log.append(operation)
            
            print(f"Peeling operation detected! Count: {self.peeling_count} (Frame: {self.frame_count}, Wire ID: {wire_id})")
            return True
        
        return False
    
    def cleanup_old_wires(self):
        """Remove wires that haven't been seen recently"""
        cutoff_frame = self.frame_count - 10
        wires_to_remove = [
            wire_id for wire_id, wire_data in self.wire_states.items()
            if wire_data['last_seen_frame'] < cutoff_frame
        ]
        
        for wire_id in wires_to_remove:
            del self.wire_states[wire_id]
    
    def process_detections(self, results):
        """Process YOLO detection results"""
        detections = []
        
        if results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                if confidence >= self.confidence:
                    class_name = self.class_names[class_id]
                    
                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'class': class_name,
                        'class_id': class_id
                    }
                    detections.append(detection)
                    self.total_detections[class_name] += 1
        
        return detections
    
    def draw_monitoring_interface(self, frame):
        """Draw the monitoring interface on the frame"""
        height, width = frame.shape[:2]
        
        # Determine box color
        if self.box_flash_counter > 0:
            box_color = self.detection_box_color
            self.box_flash_counter -= 1
        else:
            box_color = self.monitoring_box_color
        
        # Draw machine monitoring area
        if self.machine_area:
            x1, y1, x2, y2 = [int(coord) for coord in self.machine_area]
            # Ensure coordinates are within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            # Draw thick monitoring box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)
            
            # Add "MONITORING" label
            label = "MONITORING AREA"
            if self.box_flash_counter > 0:
                label = "PEELING DETECTED!"
            
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            # Draw label background
            label_x = x1
            label_y = max(y1 - 10, label_height + 10)
            cv2.rectangle(
                frame,
                (label_x, label_y - label_height - 5),
                (label_x + label_width + 10, label_y + 5),
                box_color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (label_x + 5, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        # Draw counter in top-left
        counter_text = f"Count: {self.peeling_count}"
        counter_font = cv2.FONT_HERSHEY_SIMPLEX
        counter_scale = 1.2
        counter_thickness = 3
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            counter_text, counter_font, counter_scale, counter_thickness
        )
        
        # Draw counter background
        cv2.rectangle(
            frame,
            (10, 10),
            (text_width + 30, text_height + 30),
            (0, 0, 0),
            -1
        )
        
        # Draw counter border
        counter_color = self.detection_box_color if self.box_flash_counter > 0 else (255, 255, 255)
        cv2.rectangle(
            frame,
            (10, 10),
            (text_width + 30, text_height + 30),
            counter_color,
            2
        )
        
        # Draw counter text
        cv2.putText(
            frame,
            counter_text,
            (25, text_height + 20),
            counter_font,
            counter_scale,
            counter_color,
            counter_thickness
        )
        
        # Update cooldown counters
        if self.peeling_cooldown > 0:
            self.peeling_cooldown -= 1
        
        # Update per-wire cooldowns
        for wire_id in list(self.wire_cooldown.keys()):
            if self.wire_cooldown[wire_id] > 0:
                self.wire_cooldown[wire_id] -= 1
            else:
                del self.wire_cooldown[wire_id]
        
        return frame
    
    def process_video(self, input_video_path, output_video_path=None, show_preview=False, save_stats=True):
        """
        Process video with wire peeling counting
        
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
            
            # Try different codecs for better compatibility
            codecs_to_try = [('mp4v', '.mp4'), ('XVID', '.avi'), ('MJPG', '.avi')]
            
            for codec_name, ext in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec_name)
                    if not str(output_path).endswith(ext):
                        output_path = output_path.with_suffix(ext)
                    
                    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                    
                    if out.isOpened():
                        print(f"Output video: {output_path} (codec: {codec_name})")
                        break
                    else:
                        out.release()
                        out = None
                except Exception as e:
                    if out:
                        out.release()
                        out = None
        
        # Processing loop
        start_time = time.time()
        
        print("\nProcessing video for wire peeling counting...")
        print("Press 'q' to quit preview, 'p' to pause")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Reached end of video")
                    break
                
                self.frame_count += 1
                
                # Run YOLO inference
                results = self.model(
                    frame,
                    conf=self.confidence,
                    iou=self.iou_threshold,
                    device=self.device,
                    verbose=False
                )
                
                # Process detections
                detections = self.process_detections(results)
                
                # Update machine area
                self.update_machine_area(detections)
                
                # Track wires and count operations
                self.track_wires(detections)
                
                # Draw monitoring interface
                annotated_frame = self.draw_monitoring_interface(frame.copy())
                
                # Save frame if output video specified
                if out:
                    out.write(annotated_frame)
                
                # Show preview if requested
                if show_preview:
                    display_frame = annotated_frame
                    if width > 1280:
                        scale = 1280 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        display_frame = cv2.resize(annotated_frame, (new_width, new_height))
                    
                    cv2.imshow('Wire Peeling Counter', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nStopping video processing...")
                        break
                    elif key == ord('p'):
                        print("Paused. Press any key to continue...")
                        cv2.waitKey(0)
                
                # Print progress
                if self.frame_count % 30 == 0 or self.frame_count == total_frames:
                    progress = (self.frame_count / total_frames) * 100
                    elapsed_time = time.time() - start_time
                    avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                    
                    print(f"Progress: {progress:.1f}% | Frame: {self.frame_count}/{total_frames} | "
                          f"FPS: {avg_fps:.1f} | Peeling Count: {self.peeling_count}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            if show_preview:
                cv2.destroyAllWindows()
            
            # Print final results
            self.print_final_results()
            
            # Save statistics if requested
            if save_stats and output_video_path:
                self.save_detailed_stats(output_video_path)
    
    def print_final_results(self):
        """Print final counting results"""
        print("\n" + "=" * 60)
        print("WIRE PEELING COUNTING RESULTS")
        print("=" * 60)
        print(f"Total Peeling Operations Detected: {self.peeling_count}")
        print(f"Total Frames Processed: {self.frame_count}")
        print(f"Machine Area Detected: {'Yes' if self.machine_area else 'No'}")
        
        if self.machine_area:
            x1, y1, x2, y2 = self.machine_area
            area_width = x2 - x1
            area_height = y2 - y1
            print(f"Machine Monitoring Area: {area_width}x{area_height} pixels")
        
        print("\nTotal Detections by Class:")
        for class_name, count in self.total_detections.items():
            print(f"  {class_name}: {count}")
        
        print(f"\nActive Wire Trackers: {len(self.wire_states)}")
        
        if self.operations_log:
            print("\nPeeling Operations Timeline:")
            for i, op in enumerate(self.operations_log, 1):
                frame = op['frame']
                timestamp = time.strftime('%H:%M:%S', time.localtime(op['timestamp']))
                print(f"  {i}. Frame {frame} at {timestamp}")
        
        print("=" * 60)
    
    def save_detailed_stats(self, output_video_path):
        """Save detailed statistics to JSON file"""
        output_path = Path(output_video_path)
        stats_path = output_path.parent / f"{output_path.stem}_peeling_stats.json"
        
        stats = {
            'summary': {
                'total_peeling_operations': self.peeling_count,
                'total_frames_processed': self.frame_count,
                'machine_area_detected': self.machine_area is not None,
                'machine_area_coordinates': self.machine_area,
                'processing_timestamp': datetime.now().isoformat()
            },
            'detection_counts': self.total_detections,
            'operations_log': self.operations_log,
            'parameters': {
                'confidence_threshold': self.confidence,
                'iou_threshold': self.iou_threshold,
                'machine_area_buffer': self.machine_area_buffer,
                'max_wire_distance': self.max_wire_distance,
                'peeling_cooldown_frames': self.peeling_cooldown_frames
            }
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"Detailed statistics saved to: {stats_path}")


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Wire Peeling Counter')
    parser.add_argument('--model', required=True, help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--input', required=True, help='Path to input video')
    parser.add_argument('--output', help='Path to output video (optional)')
    parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'], help='Device to use')
    parser.add_argument('--preview', action='store_true', help='Show real-time preview')
    parser.add_argument('--no-stats', action='store_true', help='Don\'t save statistics')
    
    args = parser.parse_args()
    
    # Initialize counter
    counter = WirePeelingCounter(
        model_path=args.model,
        confidence=args.confidence,
        iou_threshold=args.iou,
        device=args.device
    )
    
    # Process video
    counter.process_video(
        input_video_path=args.input,
        output_video_path=args.output,
        show_preview=args.preview,
        save_stats=not args.no_stats
    )


# Example usage for direct execution
if __name__ == "__main__":
    # For interactive use (Jupyter/Kaggle)
    if (1 == 2):  # Change this condition to True for testing in Jupyter/Kaggle
        # Example configuration
        MODEL_PATH = "/kaggle/working/yolo_training_output/train/weights/best.pt"
        INPUT_VIDEO = "/kaggle/input/your-video-dataset/test_video.mp4"
        OUTPUT_VIDEO = "/kaggle/working/peeling_count_output.mp4"
        
        # Initialize counter
        counter = WirePeelingCounter(
            model_path=MODEL_PATH,
            confidence=0.25,
            iou_threshold=0.45,
            device='auto'
        )
        
        # Process video
        print("Starting wire peeling counting...")
        counter.process_video(
            input_video_path=INPUT_VIDEO,
            output_video_path=OUTPUT_VIDEO,
            show_preview=False,
            save_stats=True
        )
        
    else:
        # Command line usage
        main()

        """ask Claude: 
           my question:
           if during the frame processing, the yolo detects, intact_wire, and then in the next frame we get peeled_wire, then we raise the counter by 1, is that right? if yes, then what do we need the cooling period for?
        """