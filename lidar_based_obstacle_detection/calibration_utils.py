import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def inverse_rigid_trans(Tr):
    """
    Inverse a rigid body transform matrix (3x4 as [R|t]) to [R'|-R't; 0|1].
    Args:
        Tr (np.ndarray): 4x4 transformation matrix.
    Returns:
        np.ndarray: Inverted 4x4 transformation matrix.
    """
    inv_Tr = np.zeros_like(Tr)
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = -np.dot(np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    inv_Tr[3, 3] = 1
    return inv_Tr

def transform_points(points, transformation):
    """
    Apply a transformation matrix to 3D points.
    Args:
        points (np.ndarray): Nx3 array of 3D points.
        transformation (np.ndarray): 4x4 transformation matrix.
    Returns:
        np.ndarray: Transformed Nx3 points.
    """
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))  # Convert to homogeneous coordinates
    points_transformed = np.dot(points_hom, transformation.T)
    return points_transformed[:, :3]

def compute_box_3d(obj, Tr_cam_to_velo):
    """
    Compute the 8 corners of a 3D bounding box in Velodyne coordinates.
    Args:
        obj (dict): Object parameters (dimensions, location, rotation_y).
        Tr_cam_to_velo (np.ndarray): Camera to Velodyne transformation matrix.
    Returns:
        np.ndarray: Array of shape (8, 3) with the 3D box corners.
    """
    l, h, w = obj["Dimensions_length"], obj["Dimensions_height"], obj["Dimensions_width"]
    x, y, z = obj["Location_x"], obj["Location_y"], obj["Location_z"],
    ry = obj["RotationY"]

    # 3D bounding box corners in object coordinates
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # Rotation matrix around yaw axis
    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])

    # Rotate and translate corners
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z

    # Transform to Velodyne coordinates
    corners_3d_velo = transform_points(corners_3d.T, Tr_cam_to_velo)
    return corners_3d_velo