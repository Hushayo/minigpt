# MiniGPT

A GPT-style language model built and trained completely from scratch: no pretrained weights, no tokenizer library, no training framework beyond PyTorch. Character-level, ~1M parameters, trained on CPU.

## Quickstart

```powershell
cd C:\Users\rosMa\Desktop\minigpt
python data\prepare.py          # download Tiny Shakespeare + build vocab
python train.py --preset small  # train (~35 min on 2 CPU cores)
python generate.py --prompt "ROMEO:" --tokens 500
```

Resume an interrupted run with `python train.py --preset small --resume`.

## What's here

| File | Purpose |
|---|---|
| `data/prepare.py` | Downloads Tiny Shakespeare, builds char vocab, writes `train.bin` / `val.bin` |
| `model.py` | Decoder-only transformer: causal self-attention, MLP blocks, LayerNorm, weight tying |
| `train.py` | AdamW + cosine LR schedule, gradient clipping, eval, best/last checkpoints |
| `generate.py` | Temperature + top-k sampling from a checkpoint |

## Presets

| Preset | Layers | Heads | Width | Context | Params | CPU time |
|---|---|---|---|---|---|---|
| `tiny` | 2 | 2 | 64 | 64 | ~0.1M | ~2 min |
| `small` | 4 | 4 | 128 | 128 | ~0.8M | ~35 min |
| `medium` | 6 | 6 | 192 | 192 | ~3.6M | ~2 hr |

Default is `small` with 4000 steps (batch 12). Checkpoints land in `checkpoints/`; `best.pt` is chosen by validation loss.

## Train on GitHub Actions

`.github/workflows/train.yml` runs the full pipeline on a GitHub-hosted runner, so training doesn't touch your PC.

```powershell
gh workflow run train.yml -f preset=small -f steps=4000 -f batch_size=12
gh run watch
```

Download the trained checkpoint and sample when it finishes:

```powershell
gh run download --name minigpt-small-<run-number>
```

Notes:

- Private repos get a 2 vCPU / 8 GB runner and 2,000 free minutes per month.
- Public repos get a 4 vCPU / 16 GB runner with unlimited standard minutes. Flip visibility with `gh repo edit --visibility public` if you don't mind the code being public.
- GPU runners require a GitHub Team/Enterprise org plan, so standard Actions stays CPU-only.
- Jobs are capped at 6 hours; the workflow times out at 350 minutes.

## Notes

- This is a base language model, not an instruction-following chatbot. It continues text in the style of its training data.
- Char-level means it predicts one character at a time; the vocabulary is just the ~65 characters in Shakespeare.
- `--temperature` controls randomness (lower = safer), `--top-k` truncates the distribution.
