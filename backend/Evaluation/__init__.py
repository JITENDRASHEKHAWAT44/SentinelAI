"""
SentinelRAG Evaluation Package.
Modular RAGAS metrics, golden test set management, and evaluation orchestrator.
"""

from backend.Evaluation.goldens import (
    GOLDENS_FILE,
    load_goldens,
    update_golden_answer,
)
from backend.Evaluation.faithfulness import evaluate_faithfulness
from backend.Evaluation.answer_relevancy import evaluate_answer_relevancy
from backend.Evaluation.context_precision import evaluate_context_precision
from backend.Evaluation.context_recall import evaluate_context_recall
from backend.Evaluation.answer_correctness import evaluate_answer_correctness
from backend.Evaluation.evaluator import (
    run_evaluation,
    get_evaluator_llm,
    get_evaluator_embeddings,
)

__all__ = [
    "GOLDENS_FILE",
    "load_goldens",
    "update_golden_answer",
    "evaluate_faithfulness",
    "evaluate_answer_relevancy",
    "evaluate_context_precision",
    "evaluate_context_recall",
    "evaluate_answer_correctness",
    "run_evaluation",
    "get_evaluator_llm",
    "get_evaluator_embeddings",
]
