"""JiuJiuBiao orchestrator: disambiguate → solve/refine → output check.

One call, full pipeline::

    import jiujiubiao

    engine = jiujiubiao.JiuJiuBiao()                      # env-configured endpoint
    result = engine.ask("Which venue had the most wins?", table=my_table)
    print(result.answer, result.mode, result.disambiguation)

Modes: ``auto`` (long tables / databases route to the agentic solver, small
tables to the critic refiner), ``solve``, or ``refine``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

import jiujiubiao.tools  # noqa: F401 - registers built-in tools
from jiujiubiao.agents.refiner import Refiner
from jiujiubiao.agents.solver import Solver
from jiujiubiao.df_utils import get_df_info, infer_dtype
from jiujiubiao.disambiguate import DisambiguationResult, Disambiguator
from jiujiubiao.llm import default_client
from jiujiubiao.normalize import is_refusal, normalize
from jiujiubiao.registry import DEFAULT_REGISTRY, ToolRegistry
from jiujiubiao.table import Table
from jiujiubiao.tools.db import Database

# ---------------------------------------------------------------------------
# Answer-type consistency gate (WTQ-100 attribution: "who" questions answered
# with bare numbers — nu-1277 class).  Surface heuristics only; a mismatch
# triggers ONE guided retry, never a silent rewrite.
# ---------------------------------------------------------------------------
_QUESTION_TYPE = [
    (re.compile(r"^\s*(who|whom|whose)\b", re.I), "entity"),
    (re.compile(r"^\s*how (many|much)\b", re.I), "number"),
]
_NUMBER = re.compile(r"^\s*-?[\d,]+(\.\d+)?\s*$")


def _expected_type(question: str) -> str | None:
    for pattern, kind in _QUESTION_TYPE:
        if pattern.search(question):
            return kind
    return None


def _is_bare_number(answer: str) -> bool:
    return bool(_NUMBER.match(answer.strip()))


def _type_mismatch_hint(question: str, answer: str) -> str | None:
    """Instruction line to append when the answer's surface type mismatches
    the question's expected type; None when consistent (or unknowable)."""

    expected = _expected_type(question)
    if expected == "entity" and _is_bare_number(answer):
        return (
            "That answer must be a person/team/entity name exactly as written "
            "in the table, not a bare number."
        )
    if expected == "number" and answer and not _is_bare_number(answer):
        return "Answer with just the count as a bare number (no units, no words)."
    return None


@dataclass
class JiuJiuBiaoResult:
    """Everything one pipeline run produced."""

    question: str
    answer: str
    norm: str
    mode: str  # "solve" | "refine"
    task: str
    disambiguation: DisambiguationResult | None = None
    steps: int = 0
    status: str = "completed"
    history: list = field(default_factory=list)
    type_retry: bool = False


class JiuJiuBiao:
    """The unified table/database question-answering pipeline."""

    def __init__(
        self,
        llm=None,
        *,
        mode: str = "auto",
        task: str = "tqa",
        long_table_rows: int = 50,
        disambiguate: bool = False,
        clarify: Callable[[str], str] | None = None,
        registry: ToolRegistry = DEFAULT_REGISTRY,
        solver_kwargs: dict | None = None,
        refiner_kwargs: dict | None = None,
    ) -> None:
        if mode not in ("auto", "solve", "refine"):
            raise ValueError("mode must be auto, solve, or refine")
        self.llm = llm if llm is not None else default_client()
        self.mode = mode
        self.task = task
        self.long_table_rows = long_table_rows
        self.registry = registry
        self.disambiguator = (
            Disambiguator(self.llm, clarify=clarify) if disambiguate else None
        )
        self._solver_kwargs = solver_kwargs or {}
        self._refiner_kwargs = refiner_kwargs or {}

    # ------------------------------------------------------------- internals
    def _pick_mode(self, df: pd.DataFrame | None, db: Database | None) -> str:
        if self.mode != "auto":
            return self.mode
        if db is not None:
            return "solve"
        if df is not None and len(df) > self.long_table_rows:
            return "solve"
        return "refine"

    def _output_check(self, answer: str, task: str) -> tuple[str, str]:
        """RAISR-style final gate: normalize; flag refusals/empties."""

        norm = normalize(answer, "tfv" if task == "tfv" else "tqa")
        if not norm or is_refusal(answer):
            return answer, ""
        return answer, norm

    # ------------------------------------------------------------------ main
    def ask(
        self,
        question: str,
        table=None,
        *,
        db: Database | None = None,
        task: str | None = None,
        caption: str = "",
        evidence: str = "",
    ) -> JiuJiuBiaoResult:
        """Answer one question over a table (any ``Table.from_obj`` input) or a Database."""

        task = task or self.task
        tbl = Table.from_obj(table) if table is not None else None
        df = infer_dtype(tbl.to_df()) if tbl is not None else None

        disambiguation = None
        if self.disambiguator is not None:
            schema = (
                get_df_info(df)
                if df is not None
                else (db.schema_text() if db is not None else "(no data)")
            )
            disambiguation = self.disambiguator.run(question, schema)
            question = disambiguation.question

        outcome = self._run_once(question, tbl, df, db, task, caption, evidence)
        outcome.disambiguation = disambiguation
        # Answer-type consistency gate: one guided retry when the surface
        # type of the answer contradicts the question type (who->number...).
        if task == "tqa" and outcome.norm:
            hint = _type_mismatch_hint(question, outcome.answer)
            if hint:
                retry = self._run_once(
                    question + "\n" + hint, tbl, df, db, task, caption, evidence
                )
                if retry.norm and _type_mismatch_hint(question, retry.answer) is None:
                    retry.disambiguation = disambiguation
                    retry.type_retry = True
                    retry.question = question
                    return retry
        return outcome

    # -------------------------------------------------------------- run once
    def _run_once(
        self,
        question: str,
        tbl: Table | None,
        df: pd.DataFrame | None,
        db: Database | None,
        task: str,
        caption: str,
        evidence: str,
    ) -> JiuJiuBiaoResult:
        mode = self._pick_mode(df, db)
        if mode == "solve":
            solver = Solver(self.llm, self.registry, **self._solver_kwargs)
            solve_task = "bird" if db is not None else ("tabfact" if task == "tfv" else "wtq")
            result = solver.solve(
                df, question, task=solve_task, caption=caption or (tbl.title if tbl else ""),
                evidence=evidence, db=db,
            )
            answer, norm = self._output_check(result.answer, task)
            return JiuJiuBiaoResult(
                question=question, answer=answer, norm=norm, mode="solve", task=task,
                steps=result.steps, status=result.status,
                history=[{"trajectory": result.trajectory, "tool_counts": result.tool_counts}],
            )
        refiner = Refiner(self.llm, **self._refiner_kwargs)
        result = refiner.run(tbl, question, task="tfv" if task == "tfv" else "tqa")
        answer, norm = self._output_check(result.answer, task)
        return JiuJiuBiaoResult(
            question=question, answer=answer, norm=norm, mode="refine", task=task,
            steps=result.iterations,
            status="completed" if norm else "incomplete", history=result.history,
        )


def ask(question: str, table=None, **kwargs) -> JiuJiuBiaoResult:
    """One-shot convenience: build a pipeline from the environment and ask."""

    pipeline_kwargs = {
        key: kwargs.pop(key)
        for key in ("mode", "task", "disambiguate", "clarify")
        if key in kwargs
    }
    return JiuJiuBiao(**pipeline_kwargs).ask(question, table, **kwargs)


__all__ = ["JiuJiuBiao", "JiuJiuBiaoResult", "ask"]
