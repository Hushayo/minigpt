"""Prepare a character-level dataset (Tiny Shakespeare or WikiText-2)."""
import argparse
import json
import os
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
WIKITEXT_BASE = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1"
WIKITEXT_FILES = {
    "train": f"{WIKITEXT_BASE}/train-00000-of-00001.parquet",
    "val": f"{WIKITEXT_BASE}/validation-00000-of-00001.parquet",
}


def download(url, path):
    if not os.path.exists(path):
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)


def load_shakespeare():
    path = os.path.join(DATA_DIR, "input.txt")
    download(SHAKESPEARE_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    n = int(len(text) * 0.9)
    return text[:n], text[n:]


def load_wikitext2():
    import pyarrow.parquet as pq

    splits = {}
    for split, url in WIKITEXT_FILES.items():
        path = os.path.join(DATA_DIR, f"wikitext2-{split}.parquet")
        download(url, path)
        table = pq.read_table(path, columns=["text"])
        splits[split] = "".join(table.column("text").to_pylist())
    return splits["train"], splits["val"]


DATASETS = {"shakespeare": load_shakespeare, "wikitext2": load_wikitext2}


def main():
    p = argparse.ArgumentParser(description="Prepare a char-level dataset")
    p.add_argument("--dataset", choices=DATASETS, default="shakespeare")
    args = p.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    train_text, val_text = DATASETS[args.dataset]()

    chars = sorted(set(train_text) | set(val_text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {str(i): c for i, c in enumerate(chars)}
    meta = {"dataset": args.dataset, "vocab_size": len(chars), "stoi": stoi, "itos": itos}
    with open(os.path.join(DATA_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    for name, text in (("train", train_text), ("val", val_text)):
        ids = np.array([stoi[c] for c in text], dtype=np.uint16)
        ids.tofile(os.path.join(DATA_DIR, f"{name}.bin"))

    print(f"{args.dataset}: train {len(train_text):,} chars | val {len(val_text):,} | vocab {len(chars)}")


if __name__ == "__main__":
    main()
