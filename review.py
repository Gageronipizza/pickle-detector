"""
review.py

training.py drops fresh downloads into staging/pickle_raw and
staging/not_pickle_raw WITHOUT trusting the search engine's labeling. Run
this to actually look at each one and decide:

  P  -> Yes, this really is a pickle          -> moved to data/pickle
  N  -> No, this is not a pickle              -> moved to data/not_pickle
  D  -> Discard (broken/irrelevant image)     -> deleted
  Space / Right Arrow -> skip, decide later
  Q  -> quit (progress is saved as you go, nothing lost)

Only images that end up in data/pickle or data/not_pickle are ever used for
training, so this is what actually controls what the model learns from.

You can run this any time, even while training.py is running in another
terminal - just don't run two review.py windows at once.
"""

import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk

STAGING_PICKLE = Path("staging/pickle_raw")
STAGING_NOT_PICKLE = Path("staging/not_pickle_raw")
FINAL_PICKLE = Path("data/pickle")
FINAL_NOT_PICKLE = Path("data/not_pickle")

MAX_DISPLAY_SIZE = (500, 500)


class ReviewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pickle Review")

        for d in (FINAL_PICKLE, FINAL_NOT_PICKLE, STAGING_PICKLE, STAGING_NOT_PICKLE):
            d.mkdir(parents=True, exist_ok=True)

        self.queue = self._build_queue()
        self.index = 0
        self.tk_img = None  # keep a reference so Tkinter doesn't garbage-collect it

        self.label_text = tk.Label(root, text="", font=("Segoe UI", 12))
        self.label_text.pack(pady=8)

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.status = tk.Label(root, text="", font=("Segoe UI", 10), fg="gray")
        self.status.pack(pady=8)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Pickle (P)", width=15, bg="#c8e6c9",
                  command=lambda: self.decide("pickle")).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="Not Pickle (N)", width=15, bg="#ffcdd2",
                  command=lambda: self.decide("not_pickle")).grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="Discard (D)", width=15, bg="#e0e0e0",
                  command=lambda: self.decide("discard")).grid(row=0, column=2, padx=5)
        tk.Button(btn_frame, text="Skip (Space)", width=15,
                  command=self.skip).grid(row=0, column=3, padx=5)

        root.bind("p", lambda e: self.decide("pickle"))
        root.bind("P", lambda e: self.decide("pickle"))
        root.bind("n", lambda e: self.decide("not_pickle"))
        root.bind("N", lambda e: self.decide("not_pickle"))
        root.bind("d", lambda e: self.decide("discard"))
        root.bind("D", lambda e: self.decide("discard"))
        root.bind("<space>", lambda e: self.skip())
        root.bind("<Right>", lambda e: self.skip())
        root.bind("q", lambda e: root.quit())
        root.bind("Q", lambda e: root.quit())

        self.show_current()

    def _build_queue(self):
        items = []
        for f in sorted(STAGING_PICKLE.glob("*")):
            items.append((f, "pickle"))
        for f in sorted(STAGING_NOT_PICKLE.glob("*")):
            items.append((f, "not_pickle"))
        return items

    def show_current(self):
        if self.index >= len(self.queue):
            self.label_text.config(text="All caught up! No more images waiting for review.")
            self.image_label.config(image="")
            self.status.config(text=f"Reviewed {len(self.queue)} images this session. "
                                     f"Run training.py to fetch more.")
            return

        path, suggested = self.queue[self.index]

        if not path.exists():
            # already moved/deleted somehow, skip it
            self.index += 1
            self.show_current()
            return

        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail(MAX_DISPLAY_SIZE)
            self.tk_img = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.tk_img)
        except Exception:
            # broken/corrupt download, auto-discard and move on
            path.unlink(missing_ok=True)
            self.index += 1
            self.show_current()
            return

        self.label_text.config(
            text=f'Downloaded while searching for: "{suggested}"  —  is this actually a pickle?'
        )
        self.status.config(
            text=f"Image {self.index + 1} of {len(self.queue)}   |   {path.name}"
        )

    def decide(self, decision):
        if self.index >= len(self.queue):
            return
        path, _ = self.queue[self.index]

        if not path.exists():
            self.index += 1
            self.show_current()
            return

        if decision == "discard":
            path.unlink(missing_ok=True)
        else:
            target_dir = FINAL_PICKLE if decision == "pickle" else FINAL_NOT_PICKLE
            existing = len(list(target_dir.glob("img_*")))
            new_path = target_dir / f"img_{existing + 1}{path.suffix}"
            path.rename(new_path)

        self.index += 1
        self.show_current()

    def skip(self):
        self.index += 1
        self.show_current()


def main():
    root = tk.Tk()
    ReviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()