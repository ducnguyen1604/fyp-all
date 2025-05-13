# confidence_weighting.py

def fuse_scores(score_rgb, score_ir, alpha=0.6):
    """
    Compute confidence-weighted fusion score.
    """
    return alpha * score_rgb + (1 - alpha) * score_ir

def fuse_labels(label_rgb, label_ir, score_rgb, score_ir):
    """
    Select the label from the modality with the higher confidence.
    """
    return label_rgb if score_rgb >= score_ir else label_ir
