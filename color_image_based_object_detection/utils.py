import os
import cv2
import random
import numpy as np
from typing import List, Dict
from matplotlib import pyplot as plt

import torch

def perform_detection_and_nms(model, image, det_conf= 0.35, nms_thresh= 0.25):
    results = model.predict(source=image, conf=det_conf)

    detections = results[0]
    boxes = detections.boxes.xyxy.cpu().numpy()  # Bounding boxes [x_min, y_min, x_max, y_max]
    class_ids = detections.boxes.cls.cpu().numpy().astype(int)  # Class IDs
    scores = detections.boxes.conf.cpu().numpy()  # Confidence scores

    # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
    # NMS requires the format [x_min, y_min, x_max, y_max] for bounding boxes
    nms_threshold = nms_thresh  # IoU threshold for NMS
    nms_indices = torch.ops.torchvision.nms(
        torch.tensor(boxes),  # Boxes in the form [x_min, y_min, x_max, y_max]
        torch.tensor(scores),  # Confidence scores
        nms_threshold
    )

    # Filter out the boxes after NMS
    filtered_boxes = np.array(boxes[nms_indices])
    filtered_class_ids = class_ids[nms_indices]
    filtered_scores = scores[nms_indices]

    return filtered_boxes, filtered_class_ids, filtered_scores

def draw_detection_output(image, detections):

    draw_image = image.copy() ## deep copy of the input image
    
    for detection in detections:
        xmin, ymin, xmax, ymax = map(int, detection["bounding_box"])
        label = f"{detection['object_name']} ({detection['confidence']:.2f})"
        color_r = random.randint(0, 255)
        color_g = random.randint(0, 255)
        color_b = random.randint(0, 255)
        
        cv2.rectangle(draw_image, (xmin, ymin), (xmax, ymax), (color_r, color_g, color_b), 2)  # Green bounding box
        cv2.putText(draw_image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (color_r, color_g, color_b), 1)

    return draw_image

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) for two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def evaluate_detections(detections: List[Dict], gt_detections: List[Dict], iou_threshold=0.5):
    """Evaluate detections against ground truth detections."""
    tp = 0
    fp = 0
    fn = 0
    tp_boxes = []
    fp_boxes = []
    fn_boxes = []

    
    matched_gt = set()
    
    for detection in detections:
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(gt_detections):
            if gt['object_name'] == 'dontcare':
                continue
            
            if gt_idx in matched_gt:
                continue
            iou = calculate_iou(detection['bounding_box'], gt['bounding_box'])
            if iou > best_iou and iou >= iou_threshold and detection['object_name'] == gt['object_name']:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_gt_idx != -1:
            tp += 1
            tp_boxes.append(detection)
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
            fp_boxes.append(detection)

    for gt_idx, gt in enumerate(gt_detections):
        if gt_idx not in matched_gt:
            fn += 1
            fn_boxes.append(gt)

    fn = len(gt_detections) - len(matched_gt)
    
    return tp, fp, fn, tp_boxes, fp_boxes, fn_boxes

def calculate_precision_recall(TP, FP, FN):
    # Calculate precision and recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return precision, recall

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