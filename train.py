"""Train the char-level GPT on CPU (or GPU if available)."""
import argparse
import json
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from model import GPT, GPTConfig

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

PRESETS = {
    # name: (n_layer, n_head, n_embd, block_size, dropout)
    "tiny": (2, 2, 64, 64, 0.0),
    "small": (4, 4, 128, 128, 0.1),
    "medium": (6, 6, 192, 192, 0.1),
}


def parse_args():
    p = argparse.ArgumentParser(description="Train MiniGPT from scratch")
    p.add_argument("--preset", choices=PRESETS, default="small")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=40)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-dir", default=os.path.join(ROOT, "checkpoints"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def load_meta():
    with open(os.path.join(DATA_DIR, "meta.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_data():
    train = np.memmap(os.path.join(DATA_DIR, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16, mode="r")
    return train, val


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device.type == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def get_lr(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / max(1, args.warmup)
    if step >= args.steps:
        return args.min_lr
    ratio = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (args.lr - args.min_lr)


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, eval_iters):
    model.eval()
    out = {}
    for split, d in data.items():
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(d, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    meta = load_meta()
    train_data, val_data = load_data()

    n_layer, n_head, n_embd, block_size, dropout = PRESETS[args.preset]
    config = GPTConfig(
        block_size=block_size,
        vocab_size=meta["vocab_size"],
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
    )
    model = GPT(config).to(device)
    optimizer = model.configure_optimizers(args.weight_decay, args.lr)

    start_step = 0
    best_val = float("inf")
    os.makedirs(args.out_dir, exist_ok=True)
    last_path = os.path.join(args.out_dir, "last.pt")
    best_path = os.path.join(args.out_dir, "best.pt")

    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        best_val = ckpt.get("best_val", best_val)
        print(f"resumed from step {start_step} (best val {best_val:.4f})", flush=True)

    print(f"device={device} preset={args.preset} params={model.num_params():,} "
          f"block={block_size} batch={args.batch_size} steps={args.steps}", flush=True)

    def save(path, step, val):
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "meta": meta,
            "step": step,
            "best_val": val,
            "args": vars(args),
        }, path)

    model.train()
    t0 = time.time()
    running = None
    for step in range(start_step, args.steps):
        lr = get_lr(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        running = loss.item() if running is None else 0.95 * running + 0.05 * loss.item()

        if (step + 1) % args.log_interval == 0:
            dt = time.time() - t0
            print(f"step {step + 1:5d}/{args.steps} | loss {running:.4f} | lr {lr:.2e} | "
                  f"{dt / args.log_interval * 1000:.0f} ms/step", flush=True)
            t0 = time.time()

        if (step + 1) % args.eval_interval == 0 or step + 1 == args.steps:
            losses = estimate_loss(model, {"train": train_data, "val": val_data},
                                   block_size, args.batch_size, device, args.eval_iters)
            print(f"  eval @ {step + 1}: train {losses['train']:.4f} | val {losses['val']:.4f}", flush=True)
            save(last_path, step + 1, best_val)
            if losses["val"] < best_val:
                best_val = losses["val"]
                save(best_path, step + 1, best_val)
                print(f"  saved new best (val {best_val:.4f}) -> {best_path}", flush=True)

    save(last_path, args.steps, best_val)
    print(f"done. best val {best_val:.4f} | best checkpoint: {best_path}", flush=True)


if __name__ == "__main__":
    main()
