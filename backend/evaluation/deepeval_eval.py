import json
import logging

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
    FactualityMetric, # type: ignore
    AnswerRelevancyMetric, # type: ignore
    ContextualPrecisionMetric, # type: ignore
    ContextualRecallMetric, # type: ignore
) # type: ignore
from deepeval.test_case import LLMTestCase # type: ignore
from deepeval import evaluate # type: ignore

from langchain_groq import ChatGroq # type: ignore

@dataclass
class EvalResult:
    question_id:        str
    question:           str
    doc_source:         str
    difficulty:         str
    section:            str

    answer:             str

    faithfulness:       Optional[float]
    answer_relevancy:   Optional[float]
    context_precision:  Optional[float]
    context_recall:     Optional[float]

    retrieved_chunk_ids: list[str]
    confidence:          float
    fallback_triggered:  bool
    is_multi_hop:        bool

class DeepEvalEvaluator:
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            api_key=api_key,
        )
        
        self.metrics = [
            FaithfulnessMetric(model=self.llm),
            AnswerRelevancyMetric(model=self.llm),
            ContextualPrecisionMetric(model=self.llm),
            ContextualRecallMetric(model=self.llm),
        ]
    
    def _build_testcase(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str 
    ) -> LLMTestCase:
        
        return LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=ground_truth,
            retrieval_contexts=contexts,
        )
    
    def evaluate_batch(
        self,
        samples: list[dict],
    ) -> dict:
        
        testcases = []
        
        for s in samples:
            tc = self._build_testcase(
                question=s["question"],
                answer=s["answer"],
                contexts=s["contexts"],
                ground_truth=s["ground_truth"],
            )
            testcases.append(tc)
        
        results = evaluate(
            testcases,
            self.metrics,
        )
        
        per_sample = []

        for sample, res in zip(samples, results):

            scores = {
                m.name: m.score
                for m in res.metrics
            }

            per_sample.append(
                EvalResult(
                    question_id=sample.get("question_id", ""),
                    question=sample["question"],
                    doc_source=sample.get("doc_source", ""),
                    difficulty=sample.get("difficulty", ""),
                    section=sample.get("section", ""),

                    answer=sample["answer"],

                    faithfulness=scores.get("FaithfulnessMetric"),
                    answer_relevancy=scores.get("AnswerRelevancyMetric"),
                    context_precision=scores.get("ContextualPrecisionMetric"),
                    context_recall=scores.get("ContextualRecallMetric"),

                    retrieved_chunk_ids=sample.get("retrieved_chunk_ids", []),
                    confidence=sample.get("confidence", 0),
                    fallback_triggered=sample.get("fallback_triggered", False),
                    is_multi_hop=sample.get("is_multi_hop", False),
                )
            )

        # aggregate
        agg = {
            "faithfulness":
                sum(r.faithfulness for r in per_sample if r.faithfulness)/len(per_sample),

            "answer_relevancy":
                sum(r.answer_relevancy for r in per_sample if r.answer_relevancy)/len(per_sample),

            "context_precision":
                sum(r.context_precision for r in per_sample if r.context_precision)/len(per_sample),

            "context_recall":
                sum(r.context_recall for r in per_sample if r.context_recall)/len(per_sample),
        }

        return agg, per_sample

    def save_results(
        self,
        aggregate,
        per_sample,
        path="results/eval_results.json"
    ):

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        output = {
            "aggregate": aggregate,
            "per_sample": [asdict(x) for x in per_sample]
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        self.logger.info(f"Saved → {path}")
    def compute_retrieval_metrics(
        self,
        retrieved_ids: list[list[str]],
        relevant_ids: list[list[str]],
        k_values: list[int] = [5, 10],
    ) -> dict:

        if len(retrieved_ids) != len(relevant_ids):
            raise ValueError("retrieved/relevant mismatch")

        metrics = {}
        n = len(retrieved_ids)

        # recall@k precision@k
        for k in k_values:
            recalls = []
            precisions = []

            for retrieved, relevant in zip(retrieved_ids, relevant_ids):
                top_k = retrieved[:k]
                rel = set(relevant)

                hits = sum(1 for x in top_k if x in rel)

                if relevant:
                    recalls.append(hits / len(rel))

                if top_k:
                    precisions.append(hits / len(top_k))

            metrics[f"recall@{k}"] = sum(recalls)/len(recalls)
            metrics[f"precision@{k}"] = sum(precisions)/len(precisions)

        # MRR
        rr = []

        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            rel = set(relevant)
            score = 0

            for rank, chunk in enumerate(retrieved, 1):
                if chunk in rel:
                    score = 1 / rank
                    break

            rr.append(score)

        metrics["mrr"] = sum(rr)/len(rr)

        # nDCG
        k = max(k_values)
        ndcgs = []

        for retrieved, relevant in zip(retrieved_ids, relevant_ids):
            rel = set(relevant)

            dcg = sum(
                1/(i+1)
                for i, c in enumerate(retrieved[:k])
                if c in rel
            )

            idcg = sum(
                1/(i+1)
                for i in range(min(len(rel), k))
            )

            ndcgs.append(dcg/idcg if idcg else 0)

        metrics[f"ndcg@{k}"] = sum(ndcgs)/len(ndcgs)

        return metrics

if __name__ == "__main__":

    evaluator = DeepEvalEvaluator()

    test_samples = [
        {
            "question": "How do I declare a request body in FastAPI?",
            "answer": "Use Pydantic BaseModel.",
            "contexts": [
                "To declare request body create BaseModel.",
                "FastAPI reads JSON body."
            ],
            "ground_truth": "Create class from BaseModel."
        }
    ]

    results = evaluator.evaluate_batch(test_samples)
    print(results)