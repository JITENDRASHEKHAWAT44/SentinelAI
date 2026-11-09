"""
SentinelRAG Comprehensive RAGAS Evaluator.
Orchestrates end-to-end evaluation of the RAG pipeline across golden samples.
Metrics:
- Faithfulness (Factual fidelity against retrieved context)
- Answer Relevancy (Directness and commitment of answer to query)
- Context Precision (Signal-to-noise ratio in ranking of retrieved chunks)
- Context Recall (Coverage of ground truth information by retrieved context)
- Answer Correctness (Factual & semantic alignment with ground truth)
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure compatibility and environment
import backend.Evaluation.compat  # noqa: F401

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.evaluation import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from backend.Evaluation.goldens import load_goldens, GOLDENS_FILE
from backend.retrievel.retriever import retrieve_context
from backend.generation.generation import generate_answer

logger = logging.getLogger("SentinelRAG.Evaluation")


def get_evaluator_llm() -> LangchainLLMWrapper:
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


def get_evaluator_embeddings() -> LangchainEmbeddingsWrapper:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY not found in environment.")
    emb = NVIDIAEmbeddings(
        model="nvidia/nemotron-3.5-embed-1b",
        nvidia_api_key=api_key,
        max_batch_size=1,
    )
    return LangchainEmbeddingsWrapper(emb)


def run_evaluation(
    goldens_path: Optional[Path] = None,
    export_path: Optional[Path] = None,
    limit: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Executes the complete RAGAS evaluation pipeline:
    1. Loads ground-truth golden dataset.
    2. Runs guarded RAG pipeline for each question (retrieval + guarded answer).
    3. Evaluates all 5 core RAG metrics.
    4. Compiles scorecard and exports report.
    """
    raw_goldens = load_goldens(goldens_path)
    if limit:
        raw_goldens = raw_goldens[:limit]

    if verbose:
        print("\n" + "=" * 75)
        print(f"  STARTING SENTINEL-RAG RAGAS EVALUATION ({len(raw_goldens)} Goldens)")
        print("=" * 75)

    # 1. Run live RAG pipeline to generate answers and extract retrieved contexts
    eval_samples = []
    rag_execution_log = []

    for idx, item in enumerate(raw_goldens, start=1):
        q = item["question"]
        ref = item["ground_truth"]
        sample_id = item.get("id", f"sample_{idx}")

        if verbose:
            print(f"\n[{idx}/{len(raw_goldens)}] Running RAG for: '{q}'")

        # Retrieve real context chunks
        docs = retrieve_context(q, k=5)
        contexts = [d["content"] for d in docs] if docs else ["No context retrieved."]

        # Generate guarded answer
        rag_res = generate_answer(q, k=5, return_details=True)
        answer = rag_res["answer"] if isinstance(rag_res, dict) else str(rag_res)
        is_blocked = rag_res.get("is_blocked", False) if isinstance(rag_res, dict) else False

        if verbose:
            print(f"    -> Answer: {answer[:120]}...")
            print(f"    -> Guardrail Status: {'[BLOCKED]' if is_blocked else '[CLEAN]'}")

        eval_samples.append(
            SingleTurnSample(
                user_input=q,
                response=answer,
                retrieved_contexts=contexts,
                reference=ref,
            )
        )

        rag_execution_log.append({
            "id": sample_id,
            "question": q,
            "ground_truth": ref,
            "response": answer,
            "is_blocked": is_blocked,
            "rail_triggered": rag_res.get("rail_triggered") if isinstance(rag_res, dict) else None,
            "retrieved_contexts": contexts,
        })

    # 2. Configure RAGAS metrics & LLM / Embeddings
    eval_llm = get_evaluator_llm()
    eval_emb = get_evaluator_embeddings()

    # Configure metrics with explicit LLM and Embeddings
    faithfulness.llm = eval_llm
    answer_relevancy.llm = eval_llm
    answer_relevancy.embeddings = eval_emb
    context_precision.llm = eval_llm
    context_recall.llm = eval_llm
    answer_correctness.llm = eval_llm
    answer_correctness.embeddings = eval_emb

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ]

    dataset = EvaluationDataset(samples=eval_samples)

    if verbose:
        print("\n" + "-" * 75)
        print("  Evaluating RAGAS Metrics (Faithfulness, Relevancy, Precision, Recall, Correctness)...")
        print("-" * 75)

    eval_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_emb,
        show_progress=verbose,
    )

    df_scores = eval_result.to_pandas()

    # Calculate metric averages
    metrics_summary = {}
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]
    for col in metric_cols:
        if col in df_scores.columns:
            metrics_summary[col] = round(float(df_scores[col].mean()), 4)
        else:
            metrics_summary[col] = 0.0

    # Overall composite RAG score
    valid_scores = [v for v in metrics_summary.values() if v is not None and not (v != v)]  # drop NaNs
    composite_score = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else 0.0

    # Compile final report structure
    report = {
        "summary": {
            "total_samples": len(raw_goldens),
            "composite_ragas_score": composite_score,
            "metrics": metrics_summary,
        },
        "scorecard_per_sample": df_scores.to_dict(orient="records"),
        "rag_executions": rag_execution_log,
    }

    # Save to disk
    save_file = export_path or (Path(__file__).resolve().parent / "evaluation_report.json")
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if verbose:
        print("\n" + "=" * 75)
        print("                      SENTINEL-RAG EVALUATION SCORECARD")
        print("=" * 75)
        print(f" Total Samples Evaluated : {len(raw_goldens)}")
        print(f" Composite RAGAS Score   : {composite_score * 100:.2f}%")
        print("-" * 75)
        print(f" • Faithfulness          : {metrics_summary.get('faithfulness', 0):.4f}")
        print(f" • Answer Relevancy      : {metrics_summary.get('answer_relevancy', 0):.4f}")
        print(f" • Context Precision     : {metrics_summary.get('context_precision', 0):.4f}")
        print(f" • Context Recall        : {metrics_summary.get('context_recall', 0):.4f}")
        print(f" • Answer Correctness    : {metrics_summary.get('answer_correctness', 0):.4f}")
        print("=" * 75)
        print(f" Detailed report saved to: {save_file}")

    return report


if __name__ == "__main__":
    run_evaluation(limit=2)
