import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=None)  # weights are loaded from checkpoint
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="models/best_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]

    model = build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    ranked = sorted(zip(class_names, probs), key=lambda kv: -kv[1])
    print(f"\nPrediction for {Path(args.image).name}:")
    for name, p in ranked:
        bar = "#" * int(p * 30)
        print(f"  {name:<14} {p*100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
