"""Download Tiny Shakespeare and build a character-level tokenizer."""
import json
import os
import urllib.request

import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    txt_path = os.path.join(DATA_DIR, "input.txt")
    if not os.path.exists(txt_path):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, txt_path)

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {str(i): c for i, c in enumerate(chars)}
    meta = {"vocab_size": len(chars), "stoi": stoi, "itos": itos}
    with open(os.path.join(DATA_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    ids = np.array([stoi[c] for c in text], dtype=np.uint16)
    n = int(len(ids) * 0.9)
    ids[:n].tofile(os.path.join(DATA_DIR, "train.bin"))
    ids[n:].tofile(os.path.join(DATA_DIR, "val.bin"))
    print(f"{len(text):,} chars | vocab {len(chars)} | train {n:,} | val {len(ids) - n:,}")


if __name__ == "__main__":
    main()
