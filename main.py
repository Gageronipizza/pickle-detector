"""
main.py

Loads the model checkpoint produced by training.py, lets you point it at an
image, tells you whether it thinks it's a pickle, and lets you correct it if
it's wrong. Corrections get saved into the same data/ folders that
training.py uses, so the next time you run training.py it will learn from
your corrections too.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

CHECKPOINT = Path("pickle_model.pt")
DATA_DIR = Path("data")
PICKLE_DIR = DATA_DIR / "pickle"
NOT_PICKLE_DIR = DATA_DIR / "not_pickle"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def load_model(device):
    if not CHECKPOINT.exists():
        print("No trained model found (pickle_model.pt is missing).")
        print("Run training.py first and let it complete at least one round.")
        sys.exit(1)

    ckpt = torch.load(CHECKPOINT, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    classes = ckpt.get("classes", ["not_pickle", "pickle"])
    return model, classes


def predict(model, classes, image_path, device):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    idx = torch.argmax(probs).item()
    return classes[idx], probs[idx].item(), img


def save_correction(img, correct_label):
    target_dir = PICKLE_DIR if correct_label == "pickle" else NOT_PICKLE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(target_dir.glob("correction_*.jpg")))
    save_path = target_dir / f"correction_{existing + 1}.jpg"
    img.convert("RGB").save(save_path)
    print(f"Saved to {save_path}. training.py will pick this up next time it runs.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, classes = load_model(device)
    print(f"Model classes: {classes}")

    while True:
        raw = input("\nEnter path to an image (or 'q' to quit): ").strip()
        if raw.lower() == "q":
            break

        image_path = Path(raw.strip('"').strip("'"))
        if not image_path.exists():
            print("That file doesn't exist. Try again.")
            continue

        try:
            label, confidence, img = predict(model, classes, image_path, device)
        except Exception as e:
            print(f"Couldn't read that as an image: {e}")
            continue

        is_pickle = "pickle" in label and "not" not in label
        print(f"\nPrediction: {'PICKLE' if is_pickle else 'NOT A PICKLE'} "
              f"(confidence: {confidence * 100:.1f}%)")

        answer = input("Is that correct? (y/n): ").strip().lower()
        if answer == "n":
            correct = input("What's the correct label? (pickle / not_pickle): ").strip().lower()
            if correct in ("pickle", "not_pickle"):
                save_correction(img, correct)
            else:
                print("Didn't recognize that label ('pickle' or 'not_pickle'), skipping.")


if __name__ == "__main__":
    main()