# fuse_detections.py

from fusion.iou_matcher import match_boxes_iou
from fusion.confidence_weighting import fuse_scores, fuse_labels

def average_boxes(box1, box2):
    """
    Average coordinates of two boxes.
    """
    return [
        (box1[0] + box2[0]) / 2,
        (box1[1] + box2[1]) / 2,
        (box1[2] + box2[2]) / 2,
        (box1[3] + box2[3]) / 2,
    ]

def fuse_detections(boxes_rgb, boxes_ir, alpha=0.6, iou_thresh=0.5):
    """
    Perform late fusion between RGB and IR detections.
    Input: lists of dicts with keys: ['bbox', 'score', 'label']
    Output: fused list of dicts
    """
    fused = []

    matched_pairs = match_boxes_iou(boxes_rgb, boxes_ir, iou_thresh=iou_thresh)

    for rgb, ir in matched_pairs:
        fused_box = average_boxes(rgb['bbox'], ir['bbox'])
        fused_score = fuse_scores(rgb['score'], ir['score'], alpha=alpha)
        fused_label = fuse_labels(rgb['label'], ir['label'], rgb['score'], ir['score'])

        fused.append({
            'bbox': fused_box,
            'score': fused_score,
            'label': fused_label
        })

    return fused
