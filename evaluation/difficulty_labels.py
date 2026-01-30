from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple


Difficulty3 = Literal["easy", "medium", "hard"]


def normalize_difficulty_to_3class(raw: str) -> Difficulty3:
    """
    Map dataset difficulty strings to {easy, medium, hard}.

    Current dataset (`evaluation/miguel.json`) uses:
      - easy
      - medium
      - hard
      - easy-medium
    """
    d = (raw or "").strip().lower()
    if d == "easy":
        return "easy"
    if d == "medium":
        return "medium"
    if d == "hard":
        return "hard"
    if d in {"easy-medium", "medium-easy", "easy_medium", "easymedium"}:
        return "medium"
    raise ValueError(f"Unknown difficulty label: {raw!r}")


def load_prompt_difficulty_map(dataset_json: str | Path) -> Dict[str, Difficulty3]:
    p = Path(dataset_json)
    cases = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError(f"Dataset must be a JSON list: {p}")

    out: Dict[str, Difficulty3] = {}
    for c in cases:
        if not isinstance(c, dict):
            continue
        prompt = c.get("prompt")
        diff = c.get("difficulty")
        if isinstance(prompt, str) and prompt.strip() and isinstance(diff, str) and diff.strip():
            out[prompt.strip()] = normalize_difficulty_to_3class(diff)
    return out


def load_labeled_prompts(dataset_json: str | Path) -> List[Tuple[str, Difficulty3]]:
    m = load_prompt_difficulty_map(dataset_json)
    return sorted(m.items(), key=lambda x: x[0])


@dataclass(frozen=True)
class LabelEncoding:
    label2id: Dict[Difficulty3, int]
    id2label: Dict[int, Difficulty3]

    @staticmethod
    def default() -> "LabelEncoding":
        label2id: Dict[Difficulty3, int] = {"easy": 0, "medium": 1, "hard": 2}
        id2label: Dict[int, Difficulty3] = {v: k for k, v in label2id.items()}
        return LabelEncoding(label2id=label2id, id2label=id2label)

