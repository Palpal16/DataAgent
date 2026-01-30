from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evaluation.difficulty_labels import Difficulty3, LabelEncoding


def _softmax(x: torch.Tensor) -> torch.Tensor:
    x = x - x.max(dim=-1, keepdim=True).values
    return torch.exp(x) / torch.exp(x).sum(dim=-1, keepdim=True)


class PromptDifficultyClassifier:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.labels = LabelEncoding.default()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, prompt: str) -> Tuple[Difficulty3, float, Dict[str, float]]:
        enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)
        probs = _softmax(out.logits).squeeze(0).detach().cpu()
        pred_id = int(probs.argmax().item())
        pred = self.labels.id2label[pred_id]
        conf = float(probs[pred_id].item())
        dist = {self.labels.id2label[i]: float(probs[i].item()) for i in range(len(probs))}
        return pred, conf, dist


def _load_best_config_map(path: str | Path) -> Dict[str, object]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def choose_config(config_map: Dict[str, object], difficulty: Difficulty3) -> Dict[str, object]:
    by_diff = (config_map or {}).get("best_config_by_difficulty", {}) or {}
    if isinstance(by_diff, dict) and difficulty in by_diff and isinstance(by_diff[difficulty], dict):
        return dict(by_diff[difficulty])
    fb = (config_map or {}).get("fallback_best_config", {}) or {}
    return dict(fb) if isinstance(fb, dict) else {}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Route a prompt to the best config via difficulty classifier.")
    ap.add_argument("--prompt", required=True, help="User prompt to classify")
    ap.add_argument("--model-dir", default="models/difficulty_distilbert", help="Folder produced by train_difficulty_classifier.py")
    ap.add_argument("--config-map", default="evaluation/best_config_by_difficulty.json", help="JSON produced by build_best_config_by_difficulty.py")
    args = ap.parse_args(argv)

    clf = PromptDifficultyClassifier(args.model_dir)
    diff, conf, dist = clf.predict(args.prompt)

    cm = _load_best_config_map(args.config_map)
    cfg = choose_config(cm, diff)

    out = {
        "prompt": args.prompt,
        "predicted_difficulty": diff,
        "confidence": conf,
        "distribution": dist,
        "recommended_config": cfg,
        "yaml_overrides": {
            "two_stage_cot": bool(cfg.get("two_stage_cot", False)),
            "best_of_n": int(cfg.get("best_of_n", 1)),
            "temperature": float(cfg.get("temperature", 0.0)),
            "temperature_max": float(cfg.get("temperature_max", 0.0)),
        },
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

