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


# ---------------------------------------------------------------------------
# V2: detect → candidate mapping → clarification WITH candidates → anchored
# rewrite.  The v1 failure mode (CLAMBSQL weak types, 2026-10-10): the
# clarification question drifts from the answer's schema mapping, so the
# rewrite drifts too.  V2 grounds both steps in explicit term→column
# candidates and rewrites with the RESOLVED mapping, not free text.
# ---------------------------------------------------------------------------

DETECT_V2_PROMPT = """You are checking whether a question over a table/database is unambiguous enough to answer reliably.

Schema / table preview:
{schema}

Question: {question}

Ambiguity exists when a phrase in the question can plausibly map to MULTIPLE columns or values in the schema, when filters or aggregations are unspecified, or when the question lacks information the table cannot provide.

KEY RULE: when the schema contains several columns a phrase could mean (e.g. "continent" vs columns mainland / continent_name; "capacity" vs max_capacity / avg_capacity), the question IS ambiguous — even if one mapping seems more natural.

Examples:
Schema: country : name, mainland, continent_name, code
Question: What is the continent of Anguilla?
Ambiguous: Yes
Ambiguous phrases: continent
Clarification question: Do you mean the mainland or the continent_name column?

Schema: student : id, name, grade
Question: How many students are there?
Ambiguous: No
Ambiguous phrases: None
Clarification question: None

Schema: building : name, max_capacity, avg_capacity
Question: What is the capacity of the auditorium?
Ambiguous: Yes
Ambiguous phrases: capacity
Clarification question: Do you mean max_capacity or avg_capacity?

Schema: transaction : transaction_type, transaction_amount
Question: Show the transaction types and the total of transactions.
Ambiguous: Yes
Ambiguous phrases: total of transactions
Clarification question: Should the total be the sum of transaction_amount, or the count of transactions?

Schema: course : title, assessment_type, grader
Question: Show exams and quizzes graded by Professor Smith.
Ambiguous: Yes
Ambiguous phrases: graded by Professor Smith
Clarification question: Does "graded by Professor Smith" apply to both exams and quizzes, or only quizzes?

Schema: author : name, active_years, retired
Question: List the authors who are not retired or active before 1990.
Ambiguous: Yes
Ambiguous phrases: not retired or active before 1990
Clarification question: Does "not" apply only to "retired", or to the whole "retired or active before 1990"?

Now check:
Schema / table preview:
{schema}

Question: {question}

Answer strictly in this format:
Ambiguous: Yes/No
Ambiguous phrases: <comma-separated phrases from the question, or None>
Clarification question: <one concrete question to ask the user, or None>
"""

MAP_PROMPT = """A question over a database is ambiguous. Map each ambiguous phrase to its candidate columns/values from the schema.

Schema:
{schema}

Question: {question}
Ambiguous phrases: {phrases}

Answer with a JSON object only: {{"<phrase>": ["<candidate column or value>", ...]}} — candidates must come from the schema text, most likely first."""

ANCHORED_REWRITE_PROMPT = """Rewrite the question to remove ambiguity, anchored on the resolved mapping.

Schema / table preview:
{schema}

Original question: {question}
Clarification question: {clarification}
User answer: {answer}
Resolved mapping (the authoritative interpretation): {mapping}

Rules: the rewritten question must use the resolved mapping's schema names (or their exact natural form), be self-contained, and answerable from the schema.

Rewritten question:"""


RESOLVE_PROMPT = """A user answered a clarification question about an ambiguous phrase.

Phrase: "{phrase}"
Candidate columns/values: {candidates}
User answer: {answer}

Which candidate does the answer mean? Reply with the exact candidate text, or None if none matches."""


class DisambiguatorV2:
    """CLEAR-structured disambiguation: detect → map candidates → clarify →
    anchored rewrite.  Drop-in replacement for :class:`Disambiguator`."""

    def __init__(self, llm: Any, clarify: Callable[[str], str] | None = None) -> None:
        self.llm = llm
        self.clarify = clarify

    # ------------------------------------------------------------------ steps
    def _detect(self, question: str, schema: str) -> tuple[bool, str, str]:
        out = self.llm.chat(
            DETECT_V2_PROMPT.format(schema=schema, question=question), temperature=0.0
        )
        ambiguous = "ambiguous: yes" in out.lower()
        phrases = clarification = ""
        for line in out.splitlines():
            lower = line.lower()
            if lower.startswith("ambiguous phrases:"):
                phrases = line.split(":", 1)[1].strip()
            elif lower.startswith("clarification question:"):
                clarification = line.split(":", 1)[1].strip()
        return ambiguous, phrases, clarification

    def _map_candidates(self, question: str, phrases: str, schema: str) -> dict[str, list[str]]:
        import json
        import re

        if not phrases or phrases.lower() in ("none", "n/a"):
            return {}
        out = self.llm.chat(
            MAP_PROMPT.format(schema=schema, question=question, phrases=phrases),
            temperature=0.0,
        )
        match = re.search(r"\{.*\}", out, flags=re.S)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return {str(k): [str(v) for v in vals] for k, vals in parsed.items() if isinstance(vals, list)}

    # ------------------------------------------------------------------- main
    def run(self, question: str, schema: str) -> DisambiguationResult:
        ambiguous, phrases, clarification = self._detect(question, schema)
        if not ambiguous:
            return DisambiguationResult(question, question, False, "", "")

        candidates = self._map_candidates(question, phrases, schema)
        # The clarification question names the candidates so the user's
        # answer can be mapped back deterministically.
        if candidates and clarification:
            options = "; ".join(
                f"'{phrase}' → {' / '.join(cands)}" for phrase, cands in candidates.items()
            )
            clarification = f"{clarification} (candidates: {options})"
        elif candidates:
            clarification = "; ".join(
                f"For '{phrase}', do you mean {', or '.join(cands)}?"
                for phrase, cands in candidates.items()
            )
        if not clarification:
            return DisambiguationResult(question, question, True, "", "")
        if self.clarify is None:
            # No human channel: keep the original question, flag the ambiguity.
            return DisambiguationResult(question, question, True, clarification, "")

        answer = self.clarify(clarification)
        # Resolve each phrase against its candidates using the user's answer
        # (never default to the first candidate — that was the v2 drift bug).
        resolved_parts: list[str] = []
        for phrase, cands in candidates.items():
            if not cands:
                continue
            if len(cands) == 1:
                chosen = cands[0]
            else:
                chosen_raw = self.llm.chat(
                    RESOLVE_PROMPT.format(
                        phrase=phrase, candidates=", ".join(cands), answer=answer
                    ),
                    temperature=0.0,
                ).strip().strip('"').strip("'")
                chosen = chosen_raw if chosen_raw in cands else cands[0]
            resolved_parts.append(f"'{phrase}' means '{chosen}'")
        # Anchor the rewrite on the RESOLVED mapping so the model cannot
        # drift from the schema (v1/v2-early failure mode on CLAMBSQL).
        mapping = "; ".join(resolved_parts) or phrases
        rewritten = self.llm.chat(
            ANCHORED_REWRITE_PROMPT.format(
                schema=schema, question=question, clarification=clarification,
                answer=answer, mapping=mapping,
            ),
            temperature=0.0,
        ).strip()
        # The rewrite may prefix boilerplate ("Rewritten question: ...").
        for prefix in ("rewritten question:", "**rewritten question:**"):
            if rewritten.lower().startswith(prefix):
                rewritten = rewritten[len(prefix):].strip().strip("*").strip()
        return DisambiguationResult(question, rewritten or question, True, clarification, answer)


__all__ = ["DisambiguationResult", "Disambiguator", "DisambiguatorV2"]
