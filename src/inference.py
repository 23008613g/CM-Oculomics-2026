# -*- coding: utf-8 -*-
"""Persistent/recurrent neovascularization research score from a fundus image.

The development endpoint is persistence/recurrence after six ranibizumab
injections versus documented regression, with NVE/NVD already present at baseline.
The legacy intolerance_risk output key is retained for compatibility; it stores
the positive-class research score, not adverse-drug intolerance or a calibrated
probability. Model computation, class order and decision threshold are unchanged.
"""
import argparse
import torch
from PIL import Image
from .model import load_dinov2
from .dataset import build_transforms


def predict(image_path: str, weights: str, device: str = None, threshold: float = 0.5):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dinov2(weights, device)   # primary backbone: DINOv2 (ViT-L/14)
    tf = build_transforms(224, train=False)
    x = tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(x), 1)[0, 1].item()
    return {
        "intolerance_risk": round(prob, 4),
        "prediction": "persistent_or_recurrent_neovascularization" if prob >= threshold else "documented_regression",
        "threshold": threshold,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Persistent/recurrent neovascularization research score; legacy intolerance_risk key is retained.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    out = predict(args.image, args.weights, threshold=args.threshold)
    for k, v in out.items():
        print(f"{k}: {v}")
