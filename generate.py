"""Sample text from a trained MiniGPT checkpoint."""
import argparse
import json
import os
import sys

import torch

from model import GPT, GPTConfig

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")


def parse_args():
    p = argparse.ArgumentParser(description="Generate text with MiniGPT")
    p.add_argument("--checkpoint", default=os.path.join(ROOT, "checkpoints", "best.pt"))
    p.add_argument("--prompt", default="ROMEO:")
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    meta = ckpt.get("meta")
    if meta is None:
        with open(os.path.join(DATA_DIR, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
    stoi = meta["stoi"]
    itos = {int(k): v for k, v in meta["itos"].items()}

    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"checkpoint step {ckpt['step']} | best val {ckpt.get('best_val', float('nan')):.4f} | "
          f"params {model.num_params():,} | data {meta.get('dataset', '?')}\n", flush=True)

    # Unknown characters are dropped so any prompt can be typed.
    prompt_ids = [stoi[c] for c in args.prompt if c in stoi]
    if not prompt_ids:
        prompt_ids = [stoi["\n"]]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)

    for i in range(args.samples):
        out = model.generate(idx, args.tokens, args.temperature, args.top_k)
        text = "".join(itos[int(t)] for t in out[0])
        print(f"--- sample {i + 1} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
