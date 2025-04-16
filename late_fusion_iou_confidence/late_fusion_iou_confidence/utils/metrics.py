# metrics.py

import time
import torch

def measure_inference_time(model, dataloader, device='cuda'):
    """
    Measure average inference time per image.
    """
    model.eval()
    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            start = time.time()
            _ = model(images)
            end = time.time()
            total_time += (end - start)
            total_images += images.size(0)

    return total_time / total_images

def calculate_precision_recall(gt_boxes, pred_boxes, iou_thresh=0.5):
    """
    Simple precision & recall at a single IoU threshold.
    Boxes are dicts with ['bbox', 'label'] and ['bbox', 'label', 'score'] respectively.
    """
    matched = set()
    tp, fp = 0, 0

    for pred in pred_boxes:
        best_iou = 0
        match_id = -1
        for i, gt in enumerate(gt_boxes):
            if gt["label"] != pred["label"] or i in matched:
                continue
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                match_id = i
        if best_iou >= iou_thresh:
            tp += 1
            matched.add(match_id)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return precision, recall
