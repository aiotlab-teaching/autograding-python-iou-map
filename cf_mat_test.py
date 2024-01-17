import os
import numpy as np
from obj_det_metrics import calc_iou, ConfusionMatrix

def read_yolo_labels(filename):
    detections = []
    if os.path.exists(filename):
        with open(filename, 'r') as fp:
            lines = fp.readlines();
            for line in lines:
                detect = line.strip().split(' ')
                detections.append(detect)
    else:
        print("Warning! {} not exist! Return empty detection numpy array.".format(filename))

    return np.array(detections).astype(np.float32)

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y

def test_confusion_matrix():
    # We have 6 classes ['cone','delineator','jersey barrier','curve mirror','transformer box','fence']
    cf_mat = ConfusionMatrix(nc=6, conf=0.5, iou_thres=0.4)

    DETECT_DIR = 'labels/detect'
    GT_DIR = 'labels/gt'
    IM_WIDTH = 640; IM_HEIGHT = 640


    for item in os.listdir(DETECT_DIR):
        detect_file = os.path.join(DETECT_DIR, item).replace('\\','/') # Force to use Linux-style path
        if os.path.isfile(detect_file):

            detections = read_yolo_labels(detect_file)
            # Convert YOLO bbox format into 
            # upper-left (x1,y1) and lower-right point (x2,y2) format
            if len(detections) > 0:
                box_xyxy = xywhn2xyxy(detections[:,1:5], IM_WIDTH, IM_HEIGHT)
                detect_xyxy = np.concatenate([box_xyxy, detections[:,-1:], detections[:,0:1]], axis=1)
            else:
                detect_xyxy = None

            filename = os.path.basename(detect_file)
            gt_file = os.path.join(GT_DIR, filename).replace('\\','/')
            gt_labels = read_yolo_labels(gt_file)
            if len(gt_labels) > 0:
                box_xyxy = xywhn2xyxy(gt_labels[:,1:5], IM_WIDTH, IM_HEIGHT)
                gt_xyxy = np.concatenate([gt_labels[:,0:1], box_xyxy], axis=1)

            cf_mat.process_batch(detect_xyxy, gt_xyxy)

    assert cf_mat.matrix[0,0] == 311
    assert cf_mat.matrix[1,1] == 238
    assert cf_mat.matrix[2,2] == 26

