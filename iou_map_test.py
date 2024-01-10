import os
from obj_det_metrics import calc_iou, calc_mAP

def test_iou():
    
    assert calc_iou([0, 0, 100, 100], [50, 50, 100, 100]) == 0.25
    assert calc_iou([0, 50, 100, 100], [50, 50, 250, 100]) == 0.2
    assert calc_iou([0, 0, 100, 100], [50, 0, 100, 100]) == 0.5
    assert calc_iou([0, 0, 5, 5], [10, 10, 20, 20]) == 0
