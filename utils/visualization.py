import cv2
import numpy as np

def precompute_bbox_corners(meshes):
    """Helper to calculate 3D bounding box corners for all objects."""
    for obj_id, mesh_data in meshes.items():
        vertices = mesh_data['vertices']
        min_xyz = np.min(vertices, axis=0)
        max_xyz = np.max(vertices, axis=0)

        # Create the 8 corners of the 3D box
        corners = np.array([
            [min_xyz[0], min_xyz[1], min_xyz[2]],
            [max_xyz[0], min_xyz[1], min_xyz[2]],
            [max_xyz[0], max_xyz[1], min_xyz[2]],
            [min_xyz[0], max_xyz[1], min_xyz[2]],
            [min_xyz[0], min_xyz[1], max_xyz[2]],
            [max_xyz[0], min_xyz[1], max_xyz[2]],
            [max_xyz[0], max_xyz[1], max_xyz[2]],
            [min_xyz[0], max_xyz[1], max_xyz[2]]
        ], dtype=np.float32)

        meshes[obj_id]['bbox_3d'] = corners
    return meshes

def draw_pose(img, K, R, t, bbox_3d, label="", color=(0, 255, 255)):
    """Helper to draw axes and 3D bounding box on an image using OpenCV."""
    # Define Axis Points (e.g., 10cm long)
    # Assuming units are mm (standard for LINEMOD). 10cm = 100mm.
    axis_length = 100.0
    axis_points = np.array([[0,0,0], [axis_length,0,0], [0,axis_length,0], [0,0,axis_length]], dtype=np.float32)

    # Project 3D points to 2D image plane
    rvec, _ = cv2.Rodrigues(R)
    img_axis_points, _ = cv2.projectPoints(axis_points, rvec, t, K, None)
    img_bbox_points, _ = cv2.projectPoints(bbox_3d, rvec, t, K, None)
    
    img_axis_points = img_axis_points.reshape(-1, 2).astype(int)
    img_bbox_points = img_bbox_points.reshape(-1, 2).astype(int)

    # Draw Axes (X=Red, Y=Green, Z=Blue)
    try:
        origin = tuple(img_axis_points[0])
        img = cv2.line(img, origin, tuple(img_axis_points[1]), (0, 0, 255), 3) # X - Red (BGR)
        img = cv2.line(img, origin, tuple(img_axis_points[2]), (0, 255, 0), 3) # Y - Green
        img = cv2.line(img, origin, tuple(img_axis_points[3]), (255, 0, 0), 3) # Z - Blue
    except Exception as e:
        print(f"Warning: Could not draw axes. {e}")

    # Draw Bounding Box Edges
    edges = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]
    for start, end in edges:
        try:
            img = cv2.line(img, tuple(img_bbox_points[start]), tuple(img_bbox_points[end]), color, 2)
        except Exception as e:
            # Point might be outside image
            pass

    # Add Label
    if label:
        try:
            cv2.putText(img, label, (origin[0], origin[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except:
            pass

    return img