from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from .generation import estimate_tokens, openai_timeout_seconds
from .retrieval import RetrievedChunk


@dataclass
class EvalJudgeResult:
    faithfulness: float
    answer_compliance: float
    style_consistency: float
    refusal_appropriateness: float
    rationale: str
    input_tokens: int
    output_tokens: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "faithfulness": self.faithfulness,
            "answer_compliance": self.answer_compliance,
            "style_consistency": self.style_consistency,
            "refusal_appropriateness": self.refusal_appropriateness,
            "judge_rationale": self.rationale,
            "judge_input_tokens": self.input_tokens,
            "judge_output_tokens": self.output_tokens,
        }


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse judge JSON from response: {text[:500]}")
    return json.loads(match.group(0))


def build_judge_prompt(
    question: str,
    answer: str,
    contexts: Iterable[RetrievedChunk],
    expected_keywords: Iterable[str],
    expected_refusal: bool,
    actual_refusal: bool,
) -> str:
    context_block = "\n\n".join(
        f"[{idx}] doc_id={r.chunk.doc_id} chunk_id={r.chunk.chunk_id}\n{r.chunk.text}"
        for idx, r in enumerate(contexts, start=1)
    )
    return f"""
You are an independent evaluator for a RAG QA system. Score the answer using only the retrieved context and the evaluation metadata.
Return valid JSON only, with no markdown.

Score each field from 0.0 to 1.0:
- faithfulness: 1 means all substantive claims in the answer are supported by retrieved context, or the answer is a correct refusal.
- answer_compliance: 1 means the answer satisfies the user question and includes the expected concepts when applicable.
- style_consistency: 1 means the answer is concise, professional, bilingual-safe, and includes citations for non-refusal answers.
- refusal_appropriateness: 1 means refusal behavior matches expectation and safety/context sufficiency.

Question:
{question}

Expected keywords/concepts:
{json.dumps(list(expected_keywords), ensure_ascii=False)}
Expected refusal: {expected_refusal}
Actual refusal: {actual_refusal}

Retrieved context:
{context_block}

Answer:
{answer}

Return JSON in this exact schema:
{{
  "faithfulness": 0.0,
  "answer_compliance": 0.0,
  "style_consistency": 0.0,
  "refusal_appropriateness": 0.0,
  "rationale": "brief reason, max 60 words"
}}
""".strip()


class LLMJudge:
    def __init__(self, provider: str, model: str, temperature: float = 0.0):
        self.provider = provider
        self.model = model
        self.temperature = temperature

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[RetrievedChunk],
        expected_keywords: Iterable[str],
        expected_refusal: bool,
        actual_refusal: bool,
    ) -> EvalJudgeResult:
        if self.provider != "openai":
            raise RuntimeError("Final evaluation must use provider=openai. Use scripts/evaluate_mock.py for local smoke tests.")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for real LLM evaluation.")

        from openai import OpenAI

        prompt = build_judge_prompt(
            question=question,
            answer=answer,
            contexts=contexts,
            expected_keywords=expected_keywords,
            expected_refusal=expected_refusal,
            actual_refusal=actual_refusal,
        )
        client = OpenAI(timeout=openai_timeout_seconds(), max_retries=1)
        response = client.responses.create(
            model=self.model,
            input=prompt,
            temperature=self.temperature,
            max_output_tokens=300,
        )
        text = getattr(response, "output_text", "") or str(response)
        parsed = _extract_json(text)
        input_tokens = estimate_tokens(prompt)
        output_tokens = estimate_tokens(text)
        usage = getattr(response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "input_tokens", input_tokens) or input_tokens
            output_tokens = getattr(usage, "output_tokens", output_tokens) or output_tokens
        return EvalJudgeResult(
            faithfulness=_clamp_score(parsed.get("faithfulness")),
            answer_compliance=_clamp_score(parsed.get("answer_compliance")),
            style_consistency=_clamp_score(parsed.get("style_consistency")),
            refusal_appropriateness=_clamp_score(parsed.get("refusal_appropriateness")),
            rationale=str(parsed.get("rationale", ""))[:500],
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
        )
