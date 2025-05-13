# visualizer.py

import cv2

COLOR_MAP = {
    0: (255, 0, 0),    # Blue
    1: (0, 255, 0),    # Green
    2: (0, 0, 255),    # Red
    "fused": (0, 255, 255)  # Yellow
}

def draw_boxes(image, boxes, label_map=None, box_color="fused", thickness=2):
    """
    Draw bounding boxes on image.
    `boxes`: list of dicts with keys ['bbox', 'score', 'label']
    """
    for obj in boxes:
        x1, y1, x2, y2 = map(int, obj["bbox"])
        label = obj["label"]
        score = obj["score"]
        color = COLOR_MAP.get(label, COLOR_MAP.get(box_color, (255, 255, 255)))

        label_str = label_map[label] if label_map else str(label)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image, f"{label_str} {score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image

def save_image(path, image):
    cv2.imwrite(path, image)
