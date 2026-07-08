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

## 3. Download images and review them

```bash
python training.py
```

This runs forever, downloading batches of candidate pickle / non-pickle
images into `staging/` and fine-tuning a ResNet18 on whatever's already been
reviewed, saving `pickle_model.pt` after every round.

**Important: raw downloads are NOT trusted automatically.** Image search
results are noisy (jars with no pickles, unrelated "pickle" results, etc.),
so training only ever happens on images you've manually confirmed. In
another terminal, run:

```bash
python review.py
```

This opens a small window showing each newly-downloaded image one at a
time. Press:
- **P** — yes, this is really a pickle → goes into `data/pickle`
- **N** — no, this isn't a pickle → goes into `data/not_pickle`
- **D** — discard (broken or irrelevant image)
- **Space / →** — skip for now

Only images sorted into `data/pickle` / `data/not_pickle` are used for
training. Run `review.py` periodically while `training.py` keeps
downloading in the background — the more you review, the better the model
gets. Stop `training.py` any time with `Ctrl+C`; the last checkpoint is
always saved.

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
- **Web-scraped labels are noisy — that's why review.py exists.** Searching
  "pickle" will pull in irrelevant results (pickle-flavored snacks, Pickle
  Rick, empty jars, etc.), and negative-class searches can misfire too.
  Nothing gets trained on until you've confirmed it in `review.py`, so
  accuracy comes down to how much you review, not how the search engine
  happened to label things. The correction loop in `main.py` adds another
  layer of cleanup on top of that.
- **Be mindful of scrape volume.** `icrawler` hits Bing's image search;
  running `training.py` for a very long time nonstop may eventually get
  rate-limited. If that happens, just pause it for a while.
- **Scraped images are for personal experimentation**, not redistribution —
  they're not yours to publish elsewhere.
- You can delete `pickle_model.pt` and the `data/` folder any time to start
  over from scratch.