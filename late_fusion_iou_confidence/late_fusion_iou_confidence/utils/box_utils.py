# box_utils.py

def xywh_to_xyxy(box):
    """
    Convert [x_center, y_center, width, height] → [x1, y1, x2, y2]
    """
    x, y, w, h = box
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

def xyxy_to_xywh(box):
    """
    Convert [x1, y1, x2, y2] → [x_center, y_center, width, height]
    """
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]

def compute_iou(boxA, boxB):
    """
    Calculate Intersection-over-Union between two [x1, y1, x2, y2] boxes.
    """
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter_area

    return inter_area / (union + 1e-6)
