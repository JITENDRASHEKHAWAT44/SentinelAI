"""
Context Precision Metric for SentinelRAG.
Measures the signal-to-noise ratio of the retriever by evaluating whether ground-truth
relevant chunks are ranked higher in the retrieved context list.
Score: 0.0 to 1.0 (Higher indicates relevant chunks are positioned at top ranks).
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import backend.Evaluation.compat  # noqa: F401

from ragas.metrics import context_precision as ragas_context_precision
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.evaluation import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_nvidia_ai_endpoints import ChatNVIDIA

METRIC_NAME = "context_precision"


def get_evaluator_llm():
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


def evaluate_context_precision(
    samples: List[Dict[str, Any]],
    llm: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Evaluates Context Precision for a list of samples.
    Each sample dict must contain:
      - 'question' (or 'user_input'): str
      - 'reference' (or 'ground_truth'): str
      - 'retrieved_contexts' (or 'contexts'): List[str]
    """
    eval_llm = llm or get_evaluator_llm()
    ragas_context_precision.llm = eval_llm

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
        metrics=[ragas_context_precision],
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
            "question": "Which student failed the examination and in what subject?",
            "response": "Neel Parmar failed with a score of 30 in Maths.",
            "retrieved_contexts": [
                "Only Neel Parmar failed (Maths = 30, below the pass mark of 35). 14 of 15 students passed.",
                "Student Attendance Summary: S007 Neel Parmar."
            ],
            "ground_truth": "Neel Parmar failed with a score of 30 in Maths, which was below the pass mark of 35.",
        }
    ]
    res = evaluate_context_precision(sample_data)
    print(f"Context Precision Average Score: {res['average_score']:.4f}")
    for item in res["details"]:
        print(f"  Q: {item.get('user_input')}")
        print(f"  Score: {item.get('context_precision')}")
