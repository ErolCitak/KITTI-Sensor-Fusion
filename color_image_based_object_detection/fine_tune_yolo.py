import os
import yaml
import numpy as np
import shutil
from tqdm import tqdm
from PIL import Image
from pathlib import Path

from ultralytics import YOLO, YOLOWorld


def convert_label_format(label_path, image_path, class_names=None):
    """
    Converts a custom label format into YOLO label format. 

    This function takes a path to a label file and the corresponding image file, processes the label information, 
    and outputs the annotations in YOLO format. YOLO format represents bounding boxes with normalized values 
    relative to the image dimensions and includes a class ID.

    Key Parameters:
    - `label_path` (str): Path to the label file in custom format.
    - `image_path` (str): Path to the corresponding image file.
    - `class_names` (list or set, optional): A collection of class names. If not provided, 
    the function will create a set of unique class names encountered in the labels.

    Processing Details:
    1. Reads the image dimensions to normalize bounding box coordinates.
    2. Filters out labels that do not match predefined classes (e.g., car, pedestrian, etc.).
    3. Converts bounding box coordinates from the custom format to YOLO's normalized center-x, center-y, width, and height format.
    4. Updates or utilizes the provided `class_names` to assign a class ID for each annotation.

    Returns:
    - `yolo_lines` (list): List of strings, each in YOLO format (<class_id> <x_center> <y_center> <width> <height>).
    - `class_names` (set or list): Updated set or list of unique class names.

    Notes:
    - The function assumes specific indices (4 to 7) for bounding box coordinates in the input label file.
    - Normalization is based on the dimensions of the input image.
    - Class filtering is limited to a predefined set of relevant classes.
    """
    
    img = Image.open(image_path)
    img_width, img_height = img.size
    pre_defined_classes = ['car', 'pedestrian', 'truck', 'cyclist', 'person_sitting', 'van', 'tram']  # misc, dontcare
    
    # Read original label file
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    yolo_lines = []
    
    # Keep track of unique class names if not provided
    if class_names is None:
        class_names = set()
    
    for line in lines:
        parts = line.strip().split()
        class_name = parts[0].lower()

        if class_name not in pre_defined_classes:
            continue
        
        # Extract bounding box coordinates (indices 4-7)
        x1, y1, x2, y2 = map(float, parts[4:8])
        
        # Convert to YOLO format (normalized)
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        # Add class name to set if not provided
        if class_names is None:
            class_names.add(class_name)
        
        # Get class ID (if class_names is a list, use index; if set, convert to list first)
        if isinstance(class_names, set):
            class_list = list(class_names)
            class_id = class_list.index(class_name)
        else:
            class_id = class_names.index(class_name)
        
        # Create YOLO format line
        yolo_line = f"{class_id} {x_center} {y_center} {width} {height}"
        yolo_lines.append(yolo_line)
    
    return yolo_lines, class_names

def create_data_yaml(images_path, labels_path, base_path, train_ratio=0.8):
    """
    Creates a dataset directory structure with train and validation splits for YOLO format.

    This function organizes image and label files into separate training and validation directories,
    converts label files to the YOLO format, and ensures the output structure adheres to YOLO conventions.

    Key Parameters:
    - `images_path` (str): Path to the directory containing the image files.
    - `labels_path` (str): Path to the directory containing the label files in custom format.
    - `base_path` (str): Base directory where the train/val split directories will be created.
    - `train_ratio` (float, optional): Ratio of images to allocate for training (default is 0.8).

    Processing Details:
    1. **Dataset Splitting**:
    - Reads all image files from `images_path` and splits them into training and validation sets 
        based on `train_ratio`.
    2. **Directory Creation**:
    - Creates the necessary directory structure for train/val splits, including `images` and `labels` subdirectories.
    3. **Label Conversion**:
    - Uses `convert_label_format` to convert label files to YOLO format.
    - Updates a set of unique class names encountered in the labels.
    4. **File Organization**:
    - Copies image files into their respective directories (train or val).
    - Writes the converted YOLO labels into the appropriate `labels` subdirectory.

    Returns:
    - None (operates directly on the file system to organize the dataset).

    Notes:
    - The function assumes labels correspond to image files with the same name (except for the file extension).
    - Handles label conversion using a predefined set of class names, ensuring consistency.
    - Uses `shutil.copy` for images to avoid removing original files.

    Dependencies:
    - Requires `convert_label_format` to be implemented for proper label conversion.
    - Relies on `os`, `shutil`, `Path`, and `tqdm` libraries.

    Usage Example:
    ```python
    create_data_yaml(
        images_path='/path/to/images',
        labels_path='/path/to/labels',
        base_path='/output/dataset',
        train_ratio=0.8
    )
    """
    
    # Get all image files
    image_files = [f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Create train/val splits
    num_train = int(len(image_files) * train_ratio)
    train_images = image_files[:num_train]
    val_images = image_files[num_train:]
    
    # Create train and val directories
    for split in ['train', 'val']:
        Path(os.path.join(base_path, split, 'images')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(base_path, split, 'labels')).mkdir(parents=True, exist_ok=True)
    
    # Initialize class_names
    class_names = set(['car', 'pedestrian', 'truck', 'cyclist', 'person_sitting', 'van', 'tram'])
    
    # Process and move files to train/val splits
    for split, image_list in [('train', train_images), ('val', val_images)]:
        for img_file in tqdm(image_list):
            
            # Get corresponding label file
            label_file = img_file.rsplit('.', 1)[0] + '.txt'
            
            if label_file == '007204.txt':
                print('7204.txt')

            # Convert labels
            original_label_path = os.path.join(labels_path, label_file)
            original_image_path = os.path.join(images_path, img_file)
            
            yolo_lines, updated_class_names = convert_label_format(
                original_label_path,
                original_image_path,
                class_names
            )
            class_names.update(updated_class_names)
            
            # Write converted labels
            new_label_path = os.path.join(base_path, split, 'labels', label_file)
            with open(new_label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            """
            # Move image
            os.rename(
                original_image_path,
                os.path.join(base_path, split, 'images', img_file)
            )
            """
            # Copy image to its new destination
            shutil.copy(original_image_path,
                        os.path.join(base_path, split, 'images', img_file))
    
    # Create data.yaml
    data = {
        'path': base_path,
        'train': os.path.join(base_path, 'train', 'images'),
        'val': os.path.join(base_path, 'val', 'images'),
        'names': list(class_names)  # Convert set to list
    }
    
    yaml_path = os.path.join(base_path, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    
    return yaml_path

def train_yolo_world(data_yaml_path, epochs=100):
    """
    Trains a YOLOv8 model on a custom dataset.

    This function leverages the YOLOv8 framework to fine-tune a pretrained model using a specified dataset
    and training configuration.

    Key Parameters:
    - `data_yaml_path` (str): Path to the YAML file containing dataset configuration (e.g., paths to train/val splits, class names).
    - `epochs` (int, optional): Number of training epochs (default is 100).

    Processing Details:
    1. **Model Initialization**:
    - Loads the YOLOv8 medium-sized model (`yolov8m.pt`) as a base model for training.
    2. **Training Configuration**:
    - Defines training hyperparameters including image size, batch size, device, number of workers, and early stopping (`patience`).
    - Results are saved to a project directory (`yolo_runs`) with a specific run name (`fine_tuning`).
    3. **Training Execution**:
    - Initiates the training process and tracks metrics such as loss and mAP.

    Returns:
    - `results`: Training results, including metrics for evaluation and performance tracking.

    Notes:
    - Assumes that the YOLOv8 framework is properly installed and accessible via `YOLO`.
    - The dataset YAML file must include paths to the training and validation datasets, as well as class names.

    Dependencies:
    - Requires the `YOLO` class from the YOLOv8 framework.

    Usage Example:
    ```python
    results = train_yolo_world(
        data_yaml_path='path/to/data.yaml',
        epochs=50
    )
    print(results)
    """
    
    # Load YOLO World model
    model = YOLO("yolov8m.pt")
    
    # Training arguments
    args = {
        'data': data_yaml_path,
        'epochs': epochs,
        'imgsz': 640,
        'batch': 16,
        'device': 0,
        'workers': 8,
        'patience': 5,
        'save': True,
        'project': 'yolo_runs',
        'name': 'fine_tuning'
    }
    
    # Start training
    results = model.train(**args)
    return results

if __name__ == "__main__":
    
    # Initialize the folder paths
    images_path = r"C:\KITTI\left_color_images\training"
    labels_path = r"C:\KITTI\training_labels\training"
    saving_base_path = r"C:\KITTI\Yolo_FineTune"
    
    # Create data.yaml and organize dataset
    ##yaml_path = create_data_yaml(images_path, labels_path, saving_base_path, train_ratio=0.8)
    yaml_path = os.path.join(saving_base_path, 'data.yaml')
    
    # Start training
    results = train_yolo_world(yaml_path, epochs=10)