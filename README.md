# Calculating IoU and mAP for Object Detection

In this homework we will write Python code to calculate the Intersection-over-Union (IoU) and mean Average Precision (mAP) for our Kaggle competition [AR Traffic Object Detection](https://www.kaggle.com/datasets/kuantinglai/ar-traffic-objects), and evaluate the detected results. 
There are two tasks need to be completed:

## 1. Implement IoU and mAP in Python

Implement two functions: **calc_iou()** and **calc_mAP()**. PyTest will be used to verify your code. Please do not modify the code of io_map_test.py


## 2. Upload Kaggle Detection Results

A studnet needs to upload the model (**obj_det_model.pt**) and result file (**detect_result.csv**) generarted in the competition. 
GitHub Auto-grading will calculate the mAP of the detection results. The mAP should pass a threshold, like 0.25, but is changeable. 
We didn't provide the Grouth Truth here because this will reveal the answer of the Kaggle competition, so you cannot test it locally.
PyTest will only check if you have upload the two files: **obj_det_model.pt** and **detect_result.csv**.
