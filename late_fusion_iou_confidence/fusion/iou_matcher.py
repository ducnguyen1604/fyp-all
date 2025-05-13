# iou_matcher.py
import torch

def compute_iou(box1, box2):
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / (union_area + 1e-6)

def match_boxes_iou(boxes_rgb, boxes_ir, iou_thresh=0.5):
    """
    Match boxes between two modalities using IoU threshold.
    Input: list of dicts with keys: ['bbox', 'score', 'label']
    Returns: list of matched pairs: (rgb_box, ir_box)
    """
    matched = []
    used_ir = set()

    for rgb in boxes_rgb:
        best_iou = 0
        best_match = None
        for i, ir in enumerate(boxes_ir):
            if i in used_ir:
                continue
            iou = compute_iou(rgb['bbox'], ir['bbox'])
            if iou > best_iou and iou >= iou_thresh:
                best_iou = iou
                best_match = (rgb, ir, i)
        if best_match:
            matched.append((best_match[0], best_match[1]))
            used_ir.add(best_match[2])
    return matched
