"""
training.py

Runs forever (until you press Ctrl+C):
1. Downloads a fresh batch of "pickle" images and a batch of "not pickle" images
   from the web using icrawler.
2. Cleans out any corrupt/unreadable downloads.
3. Runs one training round (fine-tuning a ResNet18) over the whole dataset so far.
4. Saves a checkpoint (pickle_model.pt) after every round, so you can stop
   at any time and main.py will use whatever it has learned so far.

Requires an NVIDIA GPU + CUDA-enabled PyTorch to use your 2070 Super.
If CUDA isn't available it will fall back to CPU (much slower, but still works).
"""

import time
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from icrawler.builtin import BingImageCrawler

DATA_DIR = Path("data")
PICKLE_DIR = DATA_DIR / "pickle"
NOT_PICKLE_DIR = DATA_DIR / "not_pickle"
CHECKPOINT = Path("pickle_model.pt")

IMAGES_PER_ROUND = 30  # how many new images to fetch per class, per round

# Deliberately includes some visually similar foods (cucumber, zucchini, green
# beans, asparagus) so the model has to learn real pickle features instead of
# just "is it green and long". Mix in unrelated stuff too for variety.
NEGATIVE_QUERIES = [
    "cucumber", "zucchini", "green beans", "asparagus", "jalapeno",
    "hot dog", "sandwich", "banana", "carrot", "watermelon",
    "dog", "cat", "car", "laptop", "chair", "mountain landscape",
    "bicycle", "shoe", "coffee cup", "pizza",
]


def download_images():
    PICKLE_DIR.mkdir(parents=True, exist_ok=True)
    NOT_PICKLE_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading pickle images...")
    pickle_crawler = BingImageCrawler(storage={"root_dir": str(PICKLE_DIR)})
    pickle_crawler.crawl(keyword="pickle food jar", max_num=IMAGES_PER_ROUND)

    neg_query = random.choice(NEGATIVE_QUERIES)
    print(f"Downloading non-pickle images ('{neg_query}')...")
    neg_crawler = BingImageCrawler(storage={"root_dir": str(NOT_PICKLE_DIR)})
    neg_crawler.crawl(keyword=neg_query, max_num=IMAGES_PER_ROUND)


def clean_dataset():
    """Remove any files that aren't valid, openable images."""
    from PIL import Image

    removed = 0
    for folder in (PICKLE_DIR, NOT_PICKLE_DIR):
        if not folder.exists():
            continue
        for f in folder.glob("*"):
            try:
                img = Image.open(f)
                img.verify()
            except Exception:
                f.unlink(missing_ok=True)
                removed += 1
    if removed:
        print(f"Removed {removed} corrupt/unreadable files.")


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(str(DATA_DIR), transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
    return loader, dataset.classes


def train_one_round(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    if CHECKPOINT.exists():
        print("Found existing checkpoint, resuming training from it...")
        ckpt = torch.load(CHECKPOINT, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])

    round_num = 0
    try:
        while True:
            round_num += 1
            print(f"\n=== Round {round_num} ===")

            try:
                download_images()
            except Exception as e:
                print(f"Download error this round (continuing anyway): {e}")

            clean_dataset()

            try:
                loader, classes = get_dataloader()
            except Exception as e:
                print(f"Not enough data yet to build a dataset ({e}). Retrying shortly...")
                time.sleep(5)
                continue

            print(f"Classes: {classes} | Total images: {len(loader.dataset)}")
            loss, acc = train_one_round(model, loader, optimizer, criterion, device)
            print(f"Round {round_num}: avg loss={loss:.4f}, train accuracy={acc*100:.1f}%")

            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "classes": classes,
            }, CHECKPOINT)
            print(f"Checkpoint saved to {CHECKPOINT.resolve()}")

            time.sleep(2)

    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C). Last checkpoint is already saved on disk.")


if __name__ == "__main__":
    main()