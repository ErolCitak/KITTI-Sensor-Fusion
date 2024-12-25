import os
import numpy as np
import open3d as o3d

# Function to compute IoU between two bounding boxes
def compute_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    :param box1: open3d.cpu.pybind.geometry.AxisAlignedBoundingBox object for the first box
    :param box2: open3d.cpu.pybind.geometry.AxisAlignedBoundingBox object for the second box
    :return: IoU value (float)
    """
    # Extract min and max corners
    min1, max1 = np.array(box1.min_bound), np.array(box1.max_bound)
    min2, max2 = np.array(box2.min_bound), np.array(box2.max_bound)
    
    # Compute intersection
    intersection_min = np.maximum(min1, min2)
    intersection_max = np.minimum(max1, max2)
    intersection_dims = np.maximum(intersection_max - intersection_min, 0)
    intersection_volume = np.prod(intersection_dims)
    
    # Compute union
    volume1 = np.prod(max1 - min1)
    volume2 = np.prod(max2 - min2)
    union_volume = volume1 + volume2 - intersection_volume
    
    return intersection_volume / union_volume if union_volume > 0 else 0

# Function to evaluate metrics (TP, FP, FN)
def evaluate_metrics(ground_truth_boxes, predicted_boxes, iou_threshold=0.5):
    """
    Evaluate True Positives (TP), False Positives (FP), and False Negatives (FN).
    :param ground_truth_boxes: List of AxisAlignedBoundingBox objects for ground truth
    :param predicted_boxes: List of AxisAlignedBoundingBox objects for predictions
    :param iou_threshold: IoU threshold for a match
    :return: TP, FP, FN counts
    """
    tp, fp, fn = 0, 0, 0
    used_gt = set()  # Track matched ground-truth boxes

    for pred_box in predicted_boxes:
        max_iou = 0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(ground_truth_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                best_gt_idx = gt_idx
        if max_iou >= iou_threshold:
            if best_gt_idx not in used_gt:
                tp += 1
                used_gt.add(best_gt_idx)
            else:
                fp += 1  # Duplicate detection
        else:
            fp += 1  # No match found

    # Count unmatched ground truth boxes as false negatives
    fn = len(ground_truth_boxes) - len(used_gt)
        
    return {"TP": tp, "FP": fp, "FN": fn}


def create_open3d_bounding_boxes(gt_boxes):
    """
    Convert GT boxes to Open3D AxisAlignedBoundingBox objects.

    Parameters:
    - gt_boxes: List of dictionaries with 'min' and 'max' coordinates.
                Example: [{'min': (x_min, y_min, z_min), 'max': (x_max, y_max, z_max)}]

    Returns:
    - List of open3d.geometry.AxisAlignedBoundingBox objects.
    """
    bounding_boxes = []
    for box in gt_boxes:
        min_bound = box['min']
        max_bound = box['max']
        # Create Open3D AxisAlignedBoundingBox
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
        bounding_boxes.append(bbox)
    return bounding_boxes


def parse_label_file(label_file_path):
    """
    KITTI 3D Object Detection Label Fields:

    Each line in the label file corresponds to one object in the scene and contains 15 fields:

    1. Type (string):
    - The type of object (e.g., Car, Van, Truck, Pedestrian, Cyclist, etc.).
    - "DontCare" indicates regions to ignore during training.

    2. Truncated (float):
    - Value between 0 and 1 indicating how truncated the object is.
    - 0: Fully visible, 1: Completely truncated (partially outside the image).

    3. Occluded (integer):
    - Level of occlusion:
        0: Fully visible.
        1: Partly occluded.
        2: Largely occluded.
        3: Fully occluded (annotated based on prior knowledge).

    4. Alpha (float):
    - Observation angle of the object in the image plane, ranging from [-π, π].
    - Encodes the orientation of the object relative to the camera plane.

    5. Bounding Box (4 floats):
    - (xmin, ymin, xmax, ymax) in pixels.
    - Defines the 2D bounding box in the image plane.

    6. Dimensions (3 floats):
    - (height, width, length) in meters.
    - Dimensions of the object in the 3D world.

    7. Location (3 floats):
    - (x, y, z) in meters.
    - 3D coordinates of the object center in the camera coordinate system:
        - x: Right, y: Down, z: Forward.

    8. Rotation_y (float):
    - Rotation around the Y-axis in camera coordinates, ranging from [-π, π].
    - Defines the orientation of the object in 3D space.

    9. Score (float) [optional]:
    - Confidence score for detections (used for results, not training).

    Example Line:
    Car 0.00 0 -1.82 587.00 156.40 615.00 189.50 1.48 1.60 3.69 1.84 1.47 8.41 -1.56

    Explanation of Example:
    - Type: Car
    - Truncated: 0.00 (not truncated)
    - Occluded: 0 (fully visible)
    - Alpha: -1.82 (angle in the image plane)
    - Bounding Box: (587.00, 156.40, 615.00, 189.50) (in pixels)
    - Dimensions: (1.48, 1.60, 3.69) (height, width, length in meters)
    - Location: (1.84, 1.47, 8.41) (3D coordinates in meters)
    - Rotation_y: -1.56 (orientation in 3D space)

    Notes:
    - "DontCare" objects: Regions ignored during training and evaluation. Their bounding boxes can overlap with actual objects.
    - Camera coordinates: All 3D values are given relative to the camera coordinate system, with the camera at the origin.
    """

    labels_map = [
        'Type', 'Truncated', 'Occluded', 'Alpha',
        'BBox_xmin', 'BBox_ymin', 'BBox_xmax', 'BBox_ymax',
        'Dimensions_height', 'Dimensions_width', 'Dimensions_length',
        'Location_x', 'Location_y', 'Location_z', 'RotationY'
    ]

    labels_dtypes = [
        str, float, int, float,
        float, float, float, float,
        float, float, float,
        float, float, float, float
    ]
    parsed_labels = []  # Store the parsed label dictionaries

    with open(label_file_path, "r") as file:
        # Iterate through each line in the file
        for line in file:
            line_elements = line.strip().split()  # Split line into components
            
            # Ensure the number of elements matches the expected labels
            if len(line_elements) != len(labels_map):
                raise ValueError(f"Line does not match expected format: {line.strip()}")

            # Create a dictionary for the current line
            label_line = {
                key: dtype(value)
                for key, dtype, value in zip(labels_map, labels_dtypes, line_elements)
            }

            # Append the parsed label to the list
            parsed_labels.append(label_line)

    return parsed_labels


def parse_calib_file(calib_file_path):
    """
        Parses a calibration file to extract and organize key transformation matrices.
        
        The calibration file contains the following data:
        - P0, P1, P2, P3: 3x4 projection matrices for the respective cameras.
        - R0: 3x3 rectification matrix for aligning data points across sensors.
        - Tr_velo_to_cam: 3x4 transformation matrix from the LiDAR frame to the camera frame.
        - Tr_imu_to_velo: 3x4 transformation matrix from the IMU frame to the LiDAR frame.

        Parameters:
        calib_file_path (str): Path to the calibration file.

        Returns:
        dict: A dictionary where each key corresponds to a calibration parameter 
            (e.g., 'P0', 'R0') and its value is the associated 3x4 NumPy matrix.
        
        Process:
        1. Reads the calibration file line by line.
        2. Maps each line to its corresponding key ('P0', 'P1', etc.).
        3. Extracts numerical elements, converts them to a NumPy 3x4 matrix, 
        and stores them in a dictionary.

        Example:
        Input file line for 'P0':
        P0: 1.0 0.0 0.0 0.0  0.0 1.0 0.0 0.0  0.0 0.0 1.0 0.0
        Output dictionary:
        {
            'P0': [[1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0]]
        }
    """
    calib_keys = ['P0', 'P1', 'P2', 'P3', 'R0','Tr_velo_to_cam', 'Tr_imu_to_velo']
    calibration_matrices = dict()

    with open(calib_file_path, "r") as file:
        calib_lines = file.readlines()

        for i in range(7):
            key = calib_keys[i]
            elems = calib_lines[i].split(' ')
            elems = elems[1:] # 9 , 12

            divisor = int(len(elems) / 3)
            calib_matrix = np.zeros((3,divisor), dtype=np.float32)

            for j in range(len(elems)):
                matrix_elem = float(elems[j])
                column, row = int(j % divisor), int(j / divisor)
                calib_matrix[row][column] = matrix_elem
            
            calibration_matrices[key] = calib_matrix

    return calibration_matrices


def read_velodyne_bin(file_path):
    """
    Reads a KITTI Velodyne .bin file and returns the point cloud data as a numpy array.

    :param file_path: Path to the .bin file
    :return: Numpy array of shape (N, 4) where N is the number of points,
             and each point has (x, y, z, reflectivity)
                 
    ### For KITTI's Velodyne LiDAR point cloud, the coordinate system used is forward-right-up (FRU).
    KITTI Coordinate System (FRU):
        X-axis (Forward): Points in the positive X direction move forward from the sensor.
        Y-axis (Right): Points in the positive Y direction move to the right of the sensor.
        Z-axis (Up): Points in the positive Z direction move upward from the sensor.

    
    ### Units: All coordinates are in meters (m). A point (10, 5, 2) means:

        It is 10 meters forward.
        5 meters to the right.
        2 meters above the sensor origin.
        Reflectivity: The fourth value in KITTI’s .bin files represents the reflectivity or intensity of the LiDAR laser at that point. It is unrelated to the coordinate system but adds extra context for certain tasks like segmentation or object detection.

        Velodyne Sensor Placement:

        The LiDAR sensor is mounted on a vehicle at a specific height and offset relative to the car's reference frame.
        The point cloud captures objects relative to the sensor’s position.

    """
    # Read the binary file into a 1D numpy array
    data = np.fromfile(file_path, dtype=np.float32)
    # Reshape to (N, 4) where N is the number of points
    return data.reshape(-1, 4)