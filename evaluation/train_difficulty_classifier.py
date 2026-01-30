from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evaluation.difficulty_labels import LabelEncoding, load_labeled_prompts


class PromptDifficultyDataset(Dataset):
    def __init__(
        self,
        items: List[Tuple[str, int]],
        *,
        tokenizer,
        max_length: int,
    ) -> None:
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text, label = self.items[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        out = {k: v.squeeze(0) for k, v in enc.items()}
        out["labels"] = torch.tensor(int(label), dtype=torch.long)
        return out


def _seed_all(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split(items: List[Tuple[str, int]], *, seed: int, train_ratio: float) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    items2 = list(items)
    rnd = random.Random(seed)
    rnd.shuffle(items2)
    n_train = max(1, int(math.floor(len(items2) * float(train_ratio))))
    train = items2[:n_train]
    val = items2[n_train:] or items2[:1]  # ensure non-empty val
    return train, val


@torch.no_grad()
def _evaluate(model, dl: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        logits = out.logits
        preds = logits.argmax(dim=-1)
        labels = batch["labels"]
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
        total_loss += float(loss.item()) * int(labels.numel())
    return {
        "acc": (correct / total) if total else 0.0,
        "loss": (total_loss / total) if total else 0.0,
        "n": float(total),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fine-tune DistilBERT prompt difficulty classifier (easy/medium/hard).")
    ap.add_argument("--dataset", default="evaluation/miguel.json", help="JSON dataset containing prompt + difficulty")
    ap.add_argument("--base-model", default="distilbert-base-uncased", help="HF model id to fine-tune")
    ap.add_argument("--out-dir", default="models/difficulty_distilbert", help="Output dir to save model/tokenizer")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=128)
    args = ap.parse_args(argv)

    _seed_all(args.seed)

    labels = LabelEncoding.default()
    raw_items = load_labeled_prompts(args.dataset)
    items: List[Tuple[str, int]] = [(p, labels.label2id[d]) for p, d in raw_items]
    if len(items) < 3:
        raise SystemExit(f"Not enough training items ({len(items)}). Check dataset: {args.dataset}")

    train_items, val_items = _split(items, seed=args.seed, train_ratio=args.train_ratio)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels.label2id),
        id2label={str(i): lab for i, lab in labels.id2label.items()},
        label2id={lab: i for lab, i in labels.label2id.items()},
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_ds = PromptDifficultyDataset(train_items, tokenizer=tokenizer, max_length=args.max_length)
    val_ds = PromptDifficultyDataset(val_items, tokenizer=tokenizer, max_length=args.max_length)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    optim = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    best_val_acc = -1.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)

        val_metrics = _evaluate(model, val_dl, device)
        print(f"[epoch {epoch:02d}] val_acc={val_metrics['acc']:.3f} val_loss={val_metrics['loss']:.4f} n={int(val_metrics['n'])}")

        # Save only on STRICT improvement to avoid repeatedly re-writing the same checkpoint.
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = float(val_metrics["acc"])
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)

    # Save metadata
    meta = {
        "schema": 1,
        "base_model": args.base_model,
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "train_ratio": float(args.train_ratio),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "max_length": int(args.max_length),
        "best_val_acc": float(best_val_acc),
        "label2id": labels.label2id,
        "id2label": {str(k): v for k, v in labels.id2label.items()},
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nSaved model to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

