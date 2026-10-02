"""Optional question-disambiguation stage (CLEAR-style, lean port).

The full CLEAR framework (candidate mapping, QC/MC rule correction,
CLAMBSQL) is a research prototype; this stage keeps its practical core for
production pipelines: detect whether a question is ambiguous against the
given schema, and if so either ask a human through the injected
``clarify`` callback or rewrite the question with the LLM — before any
solving happens, so every downstream component sees a cleaner input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

DETECT_PROMPT = """You are checking whether a question over a table/database is unambiguous enough to answer reliably.

Schema / table preview:
{schema}

Question: {question}

Ambiguity exists when a phrase in the question can map to multiple columns/values, when filters or aggregations are unspecified, or when the question lacks information the table cannot provide.

Answer strictly in this format:
Ambiguous: Yes/No
Clarification question: <one concrete question to ask the user, or None>
"""

REWRITE_PROMPT = """Rewrite the question to remove ambiguity, given the clarification.

Schema / table preview:
{schema}

Original question: {question}
Clarification question: {clarification}
User answer: {answer}

Rewritten question (self-contained, unambiguous, answerable from the schema):"""


@dataclass(frozen=True, slots=True)
class DisambiguationResult:
    original: str
    question: str  # possibly rewritten
    ambiguous: bool
    clarification: str  # the question asked, "" if none
    user_answer: str  # the clarification obtained, "" if none


class Disambiguator:
    """Detect → clarify → rewrite.  Inactive unless enabled in the pipeline."""

    def __init__(self, llm: Any, clarify: Callable[[str], str] | None = None) -> None:
        self.llm = llm
        self.clarify = clarify  # human-in-the-loop callback: question -> answer

    def run(self, question: str, schema: str) -> DisambiguationResult:
        detect = self.llm.chat(
            DETECT_PROMPT.format(schema=schema, question=question), temperature=0.0
        )
        ambiguous = "ambiguous: yes" in detect.lower()
        if not ambiguous:
            return DisambiguationResult(question, question, False, "", "")
        clarification = ""
        for line in detect.splitlines():
            if line.lower().startswith("clarification question:"):
                clarification = line.split(":", 1)[1].strip()
        if not clarification or clarification.lower() in ("none", "n/a"):
            return DisambiguationResult(question, question, True, "", "")
        if self.clarify is None:
            # No human channel: keep the original question, flag the ambiguity.
            return DisambiguationResult(question, question, True, clarification, "")
        answer = self.clarify(clarification)
        rewritten = self.llm.chat(
            REWRITE_PROMPT.format(
                schema=schema, question=question, clarification=clarification, answer=answer
            ),
            temperature=0.0,
        ).strip()
        return DisambiguationResult(question, rewritten or question, True, clarification, answer)


__all__ = ["DisambiguationResult", "Disambiguator"]
