# DeepONet/metrics.py
import torch


def binary_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
):
    """
    logits: [M]
    labels: [M] in {0,1}
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).long()
    labels = labels.long()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "acc": acc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
