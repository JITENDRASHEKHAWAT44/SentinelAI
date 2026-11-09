"""
Goldens Management Module for SentinelRAG Evaluation.
Loads, validates, and updates ground truth questions and reference answers.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

GOLDENS_FILE = Path(__file__).resolve().parent / "goldens.json"


def load_goldens(file_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Load golden evaluation samples from goldens.json.
    Each sample contains:
      - id: unique identifier
      - question: the user question
      - ground_truth: reference answer
      - category: topic category
    """
    target = file_path or GOLDENS_FILE
    if not target.exists():
        raise FileNotFoundError(f"Goldens file not found at: {target}")

    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def save_goldens(goldens: List[Dict[str, Any]], file_path: Optional[Path] = None) -> None:
    """Save modified golden evaluation samples back to JSON."""
    target = file_path or GOLDENS_FILE
    with open(target, "w", encoding="utf-8") as f:
        json.dump(goldens, f, indent=2, ensure_ascii=False)


def update_golden_answer(golden_id: str, new_ground_truth: str) -> bool:
    """
    Convenience method to change the reference ground truth answer for a specific golden.
    """
    goldens = load_goldens()
    updated = False
    for item in goldens:
        if item.get("id") == golden_id:
            item["ground_truth"] = new_ground_truth
            updated = True
            break

    if updated:
        save_goldens(goldens)

    return updated


if __name__ == "__main__":
    samples = load_goldens()
    print(f"Loaded {len(samples)} golden evaluation samples from {GOLDENS_FILE.name}:")
    for s in samples:
        print(f"[{s['id']}] Q: {s['question']}")
        print(f"     A: {s['ground_truth']}\n")
