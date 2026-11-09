"""
Faithfulness Metric for SentinelRAG.
Measures the factual consistency of the generated answer against the retrieved context chunks.
Score: 0.0 to 1.0 (Higher indicates higher fidelity to retrieved facts, zero hallucination).
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Compatibility & environment initialization
import backend.Evaluation.compat  # noqa: F401

from ragas.metrics import faithfulness as ragas_faithfulness
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.evaluation import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os

METRIC_NAME = "faithfulness"


def get_evaluator_llm():
    """Initializes evaluator LLM via ChatNVIDIA."""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY not found in environment.")
    llm = ChatNVIDIA(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        nvidia_api_key=api_key,
        temperature=0.0,
        max_completion_tokens=8192,
    )
    return LangchainLLMWrapper(llm)


def evaluate_faithfulness(
    samples: List[Dict[str, Any]],
    llm: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Evaluates Faithfulness for a list of samples.
    Each sample dict must contain:
      - 'question' (or 'user_input'): str
      - 'response' (or 'answer'): str
      - 'retrieved_contexts' (or 'contexts'): List[str]
    """
    eval_llm = llm or get_evaluator_llm()
    ragas_faithfulness.llm = eval_llm

    dataset_samples = []
    for s in samples:
        dataset_samples.append(
            SingleTurnSample(
                user_input=s.get("question") or s.get("user_input", ""),
                response=s.get("response") or s.get("answer", ""),
                retrieved_contexts=s.get("retrieved_contexts") or s.get("contexts", []),
                reference=s.get("ground_truth") or s.get("reference", ""),
            )
        )

    dataset = EvaluationDataset(samples=dataset_samples)
    result = evaluate(
        dataset=dataset,
        metrics=[ragas_faithfulness],
        llm=eval_llm,
        show_progress=False,
    )

    scores = result.to_pandas()
    return {
        "metric": METRIC_NAME,
        "average_score": float(scores[METRIC_NAME].mean()) if METRIC_NAME in scores else 0.0,
        "details": scores.to_dict(orient="records"),
    }


if __name__ == "__main__":
    print(f"=== Testing Standalone {METRIC_NAME.upper()} Metric ===")
    sample_data = [
        {
            "question": "Who was the top performing student and what was their score?",
            "response": "Priya Solanki was the top performer with an average score of 95.20.",
            "retrieved_contexts": [
                "TOP PERFORMER: Priya Solanki - Average: 95.20, Grade A+, Result: Pass, Rank 1."
            ],
            "ground_truth": "Priya Solanki ranked first with 95.20 average.",
        }
    ]
    res = evaluate_faithfulness(sample_data)
    print(f"Faithfulness Average Score: {res['average_score']:.4f}")
    for item in res["details"]:
        print(f"  Q: {item.get('user_input')}")
        print(f"  Score: {item.get('faithfulness')}")
