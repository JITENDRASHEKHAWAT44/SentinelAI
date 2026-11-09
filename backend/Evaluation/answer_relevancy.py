"""
Answer Relevancy Metric for SentinelRAG.
Measures how pertinent and direct the generated answer is to the user question.
Score: 0.0 to 1.0 (Higher indicates direct, committal, non-redundant answers).
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import backend.Evaluation.compat  # noqa: F401

from ragas.metrics import answer_relevancy as ragas_answer_relevancy
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.evaluation import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

METRIC_NAME = "answer_relevancy"


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


def get_evaluator_embeddings():
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY not found in environment.")
    emb = NVIDIAEmbeddings(
        model="nvidia/nemotron-3.5-embed-1b",
        nvidia_api_key=api_key,
    )
    return LangchainEmbeddingsWrapper(emb)


def evaluate_answer_relevancy(
    samples: List[Dict[str, Any]],
    llm: Optional[Any] = None,
    embeddings: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Evaluates Answer Relevancy for a list of samples.
    Each sample dict must contain:
      - 'question' (or 'user_input'): str
      - 'response' (or 'answer'): str
    """
    eval_llm = llm or get_evaluator_llm()
    eval_emb = embeddings or get_evaluator_embeddings()

    ragas_answer_relevancy.llm = eval_llm
    ragas_answer_relevancy.embeddings = eval_emb

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
        metrics=[ragas_answer_relevancy],
        llm=eval_llm,
        embeddings=eval_emb,
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
            "question": "How many students passed the examination?",
            "response": "According to the analysis report, 14 out of 15 students passed the exam.",
            "retrieved_contexts": [
                "Total students: 15. Passed students: 14. Pass percentage: 93.33%."
            ],
            "ground_truth": "14 students passed the examination.",
        }
    ]
    res = evaluate_answer_relevancy(sample_data)
    print(f"Answer Relevancy Average Score: {res['average_score']:.4f}")
    for item in res["details"]:
        print(f"  Q: {item.get('user_input')}")
        print(f"  Score: {item.get('answer_relevancy')}")
