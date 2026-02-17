# src/inference.py
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image


def save_final_model(model, threshold, model_name="efficientnet_b0",
                     img_size=224, path="fracture_model_final.pt"):
    """
    Save trained model with metadata and optimal threshold.
    """
    torch.save({
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "threshold": float(threshold),
        "img_size": img_size,
        "note": "Threshold chosen on validation set to maximize recall with specificity constraint"
    }, path)

    print(f"Model saved at: {path}")


def load_fracture_model(ckpt_path, device):
    """
    Load trained fracture classification model and preprocessing pipeline.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    if ckpt["model_name"] == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()

        preprocess = weights.transforms()
    else:
        raise ValueError("Unsupported model type")

    threshold = float(ckpt["threshold"])
    return model, preprocess, threshold


@torch.no_grad()
def predict_fracture(image_path, model, preprocess, device, threshold):
    """
    Predict fracture probability from a single X-ray image.
    Returns:
        prob (float): probability of fracture
        pred (int): 0 (no fracture) or 1 (fracture)
    """
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    logit = model(x).squeeze(1)
    prob = torch.sigmoid(logit).item()
    pred = int(prob >= threshold)

    return prob, pred
