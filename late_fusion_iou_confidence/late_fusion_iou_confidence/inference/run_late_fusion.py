# run_late_fusion.py

import json
from fusion.fuse_detections import fuse_detections
import os

def load_predictions(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():
    preds_rgb = load_predictions("results/fused_detections/preds_rgb.json")
    preds_ir = load_predictions("results/fused_detections/preds_ir.json")

    fused_results = []

    filenames = set([p["filename"] for p in preds_rgb]) | set([p["filename"] for p in preds_ir])
    for fname in filenames:
        boxes_rgb = [b for b in preds_rgb if b["filename"] == fname]
        boxes_ir = [b for b in preds_ir if b["filename"] == fname]

        fused = fuse_detections(boxes_rgb, boxes_ir, alpha=0.6, iou_thresh=0.5)
        for box in fused:
            box["filename"] = fname
            fused_results.append(box)

    output_path = "results/fused_detections/final_fused.json"
    with open(output_path, "w") as f:
        json.dump(fused_results, f, indent=2)

    print(f"Saved final fused results to {output_path}")

if __name__ == "__main__":
    main()
