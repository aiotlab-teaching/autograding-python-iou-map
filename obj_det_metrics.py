import os
import numpy as np
import pandas as pd

def calc_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    box1 (tuple): A tuple (x1, y1, x2, y2) representing the first bounding box.
    box2 (tuple): A tuple (x1, y1, x2, y2) representing the second bounding box.

    Returns:
    float: Intersection over union (IoU) between the two bounding boxes.
    """
    # Extract coordinates from the bounding boxes
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # Calculate the intersection area
    intersection_area = max(0, min(x2, x4) - max(x1, x3)) * max(0, min(y2, y4) - max(y1, y3))

    # Calculate the areas of the individual bounding boxes
    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (x4 - x3) * (y4 - y3)

    # Calculate the union area
    union_area = area_box1 + area_box2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0

    return iou

def calc_mAP():
    """
    Calculate the Mean Average Precision (mAP) for a set of bounding box predictions.

    Returns:
    float: Mean Average Precision (mAP) value.
    """
    # Placeholder for mAP calculation logic (not provided in the initial code)

    return 0.0  # Placeholder return value

# Test the calc_iou function with sample bounding boxes
iou_result = calc_iou([0, 0, 100, 100], [50, 50, 100, 100])
print(f"IoU Result: {iou_result}")
