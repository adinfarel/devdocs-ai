import json
import logging
import instructor  # type: ignore
from groq import Groq as GroqClient
from pydantic import BaseModel

# --- SETUP LOGGING ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from deepeval.metrics import ( # type: ignore
    FaithfulnessMetric, # type: ignore
    AnswerRelevancyMetric, # type: ignore
    ContextualPrecisionMetric, # type: ignore
    ContextualRecallMetric, # type: ignore
) # type: ignore
from deepeval.test_case import LLMTestCase # type: ignore
from deepeval import evaluate # type: ignore
from deepeval.models.base_model import DeepEvalBaseLLM # type: ignore

class GroqDeepEvalLLM(DeepEvalBaseLLM):
    """
    Custom Groq wrapper untuk deepeval 3.9.5.
    """
    
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_name = model_name
        api_key = os.getenv("GROQ_API_KEY", "")
        
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        
        self._client = instructor.from_groq(
            GroqClient(api_key=api_key),
            mode=instructor.Mode.JSON
        )
    
    def load_model(self):
        return self._client
    
    def generate(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        """
        Synchronization is generated with structured output via the instructor's JSON mode.

        Arguments:
        prompt: The prompt string of the internally generated deepeval metric.
        schema: The Pydantic BaseModel class that is injected deeper
        — the instructor uses this to implement the JSON output.

        Returns:
        An example of the LLM-filled schema.
        """
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": prompt
                }
            ],
            response_model=schema,
            max_retries=2,
        )
        
        return response

    async def a_generate(
        self, prompt: str, schema: type[BaseModel]
    ) -> BaseModel:
        """
        Async version — deepeval calls this when async_mode=True.
        We wrap sync generate() because our Groq client instructor syncs.
        """
        return self.generate(prompt, schema)
    
    def get_model_name(self) -> str:
        return self.model_name

@dataclass
class EvalResult:
    """
    Evaluation results for a single test sample.
    Stores all metric scores plus metadata
    for per-difficulty and per-section analysis in the notebook.
    """
    question_id:         str
    question:            str
    doc_source:          str
    difficulty:          str
    section:             str
    answer:              str
    faithfulness:        Optional[float]
    answer_relevancy:    Optional[float]
    context_precision:   Optional[float]
    context_recall:      Optional[float]
    retrieved_chunk_ids: list[str]
    confidence:          float
    fallback_triggered:  bool
    is_multi_hop:        bool

class DeepEvalEvaluator:
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        groq_llm = GroqDeepEvalLLM()
        
        self.metrics = [
            FaithfulnessMetric(
                model=groq_llm,
                threshold=0.5,
                async_mode=False, 
            ),
            AnswerRelevancyMetric(
                model=groq_llm,
                threshold=0.5,
                async_mode=False,
            ),
            ContextualPrecisionMetric(
                model=groq_llm,
                threshold=0.5,
                async_mode=False,
            ),
            ContextualRecallMetric(
                model=groq_llm,
                threshold=0.5,
                async_mode=False,
            ),
        ]
        self.logger.info(
            f"DeepEvalEvaluator initialized — "
            f"judge: {groq_llm.get_model_name()}, "
            f"{len(self.metrics)} metrics"
        )
    
    def _build_testcase(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str 
    ) -> LLMTestCase:
        """
        Construct deepeval LLMTestCase from the output pipeline.

        Args:
            question: Original user query.
            answer: Answer that our pipeline generates.
            contexts: List of retrieved chunk content strings.
                          Deepeval needs plain strings, not dicts.
            ground_truth: Reference answer from test_questions.json.

        Returns:
            LLMTestCase is ready to be evaluated.
        """
        return LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=ground_truth,
            retrieval_context=contexts,
        )
    
    def evaluate_batch(
        self,
        samples: list[dict],
    ) -> tuple[dict, list[EvalResult]]:
        """
        Run deep evaluation for batch of samples.

        Args:
            samples: List of dicts, each dict must have:
                - question: str
                - answer: str
                - contexts: list[str] (chunk contents)
                - ground_truth: str

        Returns:
            Tuple (aggregate_scores, per_sample_results).
        """
        if not samples:
            self.logger.error("No samples to evaluate")
            return {}, []
        
        testcases = []
        valid_samples = []
        
        for s in samples:
            try:
                tc = self._build_testcase(
                    question=s["question"],
                    answer=s["answer"],
                    contexts=s["contexts"],
                    ground_truth=s["ground_truth"],
                )
                testcases.append(tc)
                valid_samples.append(s)
            except KeyError as e:
                self.logger.warning(
                    f"Sample {i} missing field {e} — skipping"
                )
        
        if not testcases:
            self.logger.error("No valid test cases to evaluate")
            return {}, []
        
        self.logger.info(
            f"Running deepeval on {len(testcases)} test cases..."
        )
        
        per_sample: list[EvalResult] = []

        for i, (sample, tc) in enumerate(zip(valid_samples, testcases)):
            evaluate(
                test_cases=[tc],
                metrics=self.metrics,
            )
            score_map: dict[str, Optional[float]] = {}
            
            for metric in self.metrics:
                name = type(metric).__name__
                score = getattr(metric, "score", None)
                score_map[name] = score
                        
            self.logger.debug(f"Sample {i} scores: {score_map}")
            
            per_sample.append(
                EvalResult(
                    question_id=sample.get("question_id", ""),
                    question=sample["question"],
                    doc_source=sample.get("doc_source", ""),
                    difficulty=sample.get("difficulty", ""),
                    section=sample.get("section", ""),

                    answer=sample["answer"],

                    faithfulness=score_map.get("FaithfulnessMetric")
                    or score_map.get("FactualityMetric"),
                    answer_relevancy=score_map.get("AnswerRelevancyMetric")
                    or score_map.get("AnswerRelevancyMetric"),
                    context_precision=score_map.get("ContextualPrecisionMetric")
                    or score_map.get("ContextualPrecisionMetric"),
                    context_recall=score_map.get("ContextualRecallMetric")
                    or score_map.get("ContextualRecallMetric"),

                    retrieved_chunk_ids=sample.get("retrieved_chunk_ids", []),
                    confidence=sample.get("confidence", 0),
                    fallback_triggered=sample.get("fallback_triggered", False),
                    is_multi_hop=sample.get("is_multi_hop", False),
                )
            )
        def safe_mean(vals: list[Optional[float]]) -> Optional[float]:
            """Mean that skip None — not crash if metric fails."""
            valid = [v for v in vals if v is not None]
            return round(sum(valid) / len(valid), 4) if valid else None
        
        # aggregate
        aggregate = {
            "faithfulness":      safe_mean([r.faithfulness      for r in per_sample]),
            "answer_relevancy":  safe_mean([r.answer_relevancy  for r in per_sample]),
            "context_precision": safe_mean([r.context_precision for r in per_sample]),
            "context_recall":    safe_mean([r.context_recall    for r in per_sample]),
        }
        
        self.logger.info(
            f"Evaluation complete:\n"
            f"  Faithfulness      : {aggregate['faithfulness']}\n"
            f"  Answer Relevancy  : {aggregate['answer_relevancy']}\n"
            f"  Context Precision : {aggregate['context_precision']}\n"
            f"  Context Recall    : {aggregate['context_recall']}"
        )

        return aggregate, per_sample

    def save_results(
        self,
        aggregate: dict,
        per_sample: list[EvalResult],
        path: str="results/eval_results.json"
    ) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "aggregate": aggregate,
            "per_sample": [asdict(x) for x in per_sample]
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved → {path}")
        
    def compute_retrieval_metrics(
        self,
        retrieved_ids: list[list[str]],
        relevant_ids: list[list[str]],
        k_values: list[int] = [5, 10],
    ) -> dict:
        """
        Compute retrieval-only metrics: Recall@k, Precision@k, MRR, nDCG.

        Separate from deepeval — measures only the retriever, not the full pipeline.
        Used in run_eval.py to populate the RQ1 and RQ2 tables in the README.

        Args:
            retrieved_ids: retrieved_ids[i] = chunk_ids of the i-th query,
            ordered by rank (index 0 = rank 1).
            relevant_ids: relevant_ids[i] = ground truth chunk_ids
            for the i-th query.
            k_values: List of k values ​​for Recall@k and Precision@k.

        Returns:
            A dict containing all metric scores.
        """
        if not retrieved_ids or not relevant_ids:
            self.logger.error("Empty retrieved or relevant ids")
            return {}

        if len(retrieved_ids) != len(relevant_ids):
            raise ValueError(
                f"Length mismatch: retrieved={len(retrieved_ids)}, "
                f"relevant={len(relevant_ids)}"
            )

        metrics = {}
        n = len(retrieved_ids)

        # recall@k precision@k
        for k in k_values:
            recalls    = []
            precisions = []

            for retrieved, relevant in zip(retrieved_ids, relevant_ids):
                top_k        = retrieved[:k]
                relevant_set = set(relevant)
                hits         = sum(1 for r in top_k if r in relevant_set)

                if relevant:
                    recalls.append(hits / len(relevant_set))
                if top_k:
                    precisions.append(hits / len(top_k))
                    
            metrics[f"recall@{k}"]    = (
                sum(recalls) / len(recalls) if recalls else 0.0
            )
            metrics[f"precision@{k}"] = (
                sum(precisions) / len(precisions) if precisions else 0.0
            )

        # MRR
        reciprocal_ranks = []

        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            relevant_set = set(relevant)
            rr           = 0.0

            for rank, chunk_id in enumerate(retrieved, 1):
                if chunk_id in relevant_set:
                    rr = 1.0 / rank
                    break

            reciprocal_ranks.append(rr)

        metrics["mrr"] = (
            sum(reciprocal_ranks) / len(reciprocal_ranks)
            if reciprocal_ranks else 0.0
        )

        # nDCG
        k_ndcg      = max(k_values)
        ndcg_scores = []

        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            relevant_set = set(relevant)
            top_k        = retrieved[:k_ndcg]

            dcg = sum(
                1.0 / (i + 1)
                for i, chunk_id in enumerate(top_k)
                if chunk_id in relevant_set
            )

            n_relevant_in_k = min(len(relevant), k_ndcg)
            idcg = sum(1.0 / (i + 1) for i in range(n_relevant_in_k))

            ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

        metrics[f"ndcg@{k_ndcg}"] = (
            sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
        )

        self.logger.info(
            "Retrieval metrics:\n" +
            "\n".join(f"  {k}: {v:.4f}" for k, v in metrics.items())
        )

        return metrics


if __name__ == "__main__":
    evaluator = DeepEvalEvaluator()

    test_samples = [
        {
            "id":           "fastapi_001",
            "question":     "How do I declare a request body in FastAPI?",
            "answer":       "Use Pydantic BaseModel and declare it as a function parameter.",
            "contexts":     [
                "To declare a request body, create a class that inherits from BaseModel.",
                "FastAPI will read the request body as JSON and validate it.",
            ],
            "ground_truth": "Create a class from Pydantic BaseModel and declare as parameter.",
            "doc_source":   "fastapi",
            "difficulty":   "single_hop",
            "section":      "tutorial",
        },
    ]

    agg, per_sample = evaluator.evaluate_batch(test_samples)

    print(f"\nAggregate scores:")
    for k, v in agg.items():
        print(f"  {k}: {v}")

    # test retrieval metrics
    retrieved = [["chunk_a", "chunk_b", "chunk_c", "chunk_d", "chunk_e"]]
    relevant  = [["chunk_a", "chunk_c"]]

    ret = evaluator.compute_retrieval_metrics(retrieved, relevant)
    print(f"\nRetrieval metrics:")
    for k, v in ret.items():
        print(f"  {k}: {v:.4f}")