import os
import xml.etree.ElementTree as ET
from pathlib import Path

def convert_voc_to_yolo(voc_dir, yolo_dir, classes_file=None):
    """
    Convert Pascal VOC format annotations to YOLOv8 format
    
    Args:
        voc_dir (str): Directory containing VOC XML files
        yolo_dir (str): Output directory for YOLO format files
        classes_file (str): Path to classes.txt file (optional)
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(yolo_dir, exist_ok=True)
    
    # Get all class names from XML files if classes_file not provided
    if classes_file is None:
        classes = get_classes_from_xml(voc_dir)
    else:
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
    
    # Create classes.txt file in output directory
    with open(os.path.join(yolo_dir, 'classes.txt'), 'w') as f:
        for cls in classes:
            f.write(f"{cls}\n")
    
    # Process each XML file
    xml_files = [f for f in os.listdir(voc_dir) if f.endswith('.xml')]
    
    for xml_file in xml_files:
        xml_path = os.path.join(voc_dir, xml_file)
        
        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        # Create corresponding YOLO file
        yolo_filename = xml_file.replace('.xml', '.txt')
        yolo_path = os.path.join(yolo_dir, yolo_filename)
        
        with open(yolo_path, 'w') as yolo_file:
            # Process each object in the XML
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                
                # Skip if class not in classes list
                if class_name not in classes:
                    print(f"Warning: Class '{class_name}' not found in classes list")
                    continue
                
                class_id = classes.index(class_name)
                
                # Get bounding box coordinates
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Convert to YOLO format (normalized center coordinates and dimensions)
                x_center = (xmin + xmax) / 2.0 / img_width
                y_center = (ymin + ymax) / 2.0 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                # Write to YOLO file
                yolo_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"Conversion completed!")
    print(f"Processed {len(xml_files)} XML files")
    print(f"Output directory: {yolo_dir}")
    print(f"Classes file: {os.path.join(yolo_dir, 'classes.txt')}")

def get_classes_from_xml(voc_dir):
    """Extract all unique class names from XML files"""
    classes = set()
    
    xml_files = [f for f in os.listdir(voc_dir) if f.endswith('.xml')]
    
    for xml_file in xml_files:
        xml_path = os.path.join(voc_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            classes.add(class_name)
    
    return sorted(list(classes))

def validate_conversion(voc_dir, yolo_dir):
    """Validate the conversion by checking file counts and sample content"""
    voc_files = [f for f in os.listdir(voc_dir) if f.endswith('.xml')]
    yolo_files = [f for f in os.listdir(yolo_dir) if f.endswith('.txt') and f != 'classes.txt']
    
    print(f"\nValidation:")
    print(f"VOC files: {len(voc_files)}")
    print(f"YOLO files: {len(yolo_files)}")
    
    if len(voc_files) == len(yolo_files):
        print("✓ File count matches")
    else:
        print("✗ File count mismatch")
    
    # Check if classes.txt exists
    classes_path = os.path.join(yolo_dir, 'classes.txt')
    if os.path.exists(classes_path):
        with open(classes_path, 'r') as f:
            classes = f.readlines()
        print(f"✓ Classes file created with {len(classes)} classes")
    else:
        print("✗ Classes file not found")

# Example usage
if __name__ == "__main__":
    # Set your directories
    voc_directory = "voc_dataset/labels"  # Directory containing XML files
    yolo_directory = "yolo_dataset/labels"  # Output directory

    # Option 1: Auto-detect classes from XML files
    convert_voc_to_yolo(voc_directory, yolo_directory)
    
    # Option 2: Use predefined classes file
    # classes_file = "path/to/classes.txt"
    # convert_voc_to_yolo(voc_directory, yolo_directory, classes_file)
    
    # Validate the conversion
    validate_conversion(voc_directory, yolo_directory)