# Pickle Classifier

## 1. Python version

Use **Python 3.10**. It's old enough that PyTorch's CUDA wheels and `icrawler`
are both fully mature and stable on it, and new enough to not be on its way
out. Avoid 3.12/3.13 for now — some ML packages lag behind on new Python
releases.

## 2. Set up the environment

```bash
# create and activate a virtual environment
python3.10 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install PyTorch with CUDA support for your 2070 Super
# (CUDA 11.8 build works well and is broadly compatible)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# install the rest
pip install -r requirements.txt
```

Check that your GPU is detected:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You'll need an up-to-date NVIDIA driver installed (the CUDA *toolkit* itself
isn't required — the `cu118` PyTorch wheel bundles what it needs).

## 3. Train the model

```bash
python training.py
```

This runs forever, downloading batches of pickle / non-pickle images and
fine-tuning a ResNet18 on them, saving `pickle_model.pt` after every round.
Leave it running for a while (the longer, the better the accuracy) and stop
it any time with `Ctrl+C` — the last checkpoint is always saved.

## 4. Classify images

```bash
python main.py
```

Give it a path to an image, and it'll tell you whether it thinks it's a
pickle. If it's wrong, say so and tell it the correct label — the image gets
saved into `data/pickle` or `data/not_pickle`, so the next `training.py` run
will learn from your correction too.

## Notes / things to know

- **This needs both classes to work.** A "pickle or not" model has to see
  plenty of non-pickle images too, or it'll just learn to say "pickle" for
  everything. `training.py` handles this for you already.
- **Web-scraped labels are noisy.** Searching "pickle" will pull in some
  irrelevant results (pickle-flavored snacks, Pickle Rick, etc.). This is
  normal — a bit of noise won't hurt much, and the correction loop in
  `main.py` helps clean things up over time.
- **Be mindful of scrape volume.** `icrawler` hits Bing's image search;
  running `training.py` for a very long time nonstop may eventually get
  rate-limited. If that happens, just pause it for a while.
- **Scraped images are for personal experimentation**, not redistribution —
  they're not yours to publish elsewhere.
- You can delete `pickle_model.pt` and the `data/` folder any time to start
  over from scratch.