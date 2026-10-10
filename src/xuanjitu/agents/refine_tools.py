"""Refine-stage tools (TableCritic family, CC-BY adapted).

Each tool implements::

    generate(ctx, feedback) -> list[Candidate]      (n candidates)
    critic(ctx, candidate)  -> (need_refinement, suggestions, critic_text)

``run_python`` executes LLM-written code against the table in a separate
process (stdlib-only sandbox; not a security boundary against hostile code).
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any

from xuanjitu.normalize import (
    extract_answer,
    extract_code,
    extract_json,
    normalize,
    parse_critic,
)
from xuanjitu.prompts import refine as P

REFINE_TOOL_LIBRARY: dict[str, type] = {}


def register_refine_tool(name: str):
    def deco(cls):
        cls.name = name
        REFINE_TOOL_LIBRARY[name] = cls
        return cls

    return deco


@dataclass
class Candidate:
    tool: str
    answer: str  # raw answer
    norm: str  # normalized answer (for voting)
    trace: str  # reasoning / code, fed to the critic
    extra: dict = field(default_factory=dict)


@dataclass
class RefineContext:
    table: Any  # xuanjitu.table.Table
    question: str
    task: str  # 'tqa' | 'tfv'

    @property
    def fmt(self) -> dict:
        return dict(
            task_desc=P.TASK_DESC[self.task],
            qname=P.Q_NAME[self.task],
            qname_l=P.Q_NAME[self.task].lower(),
            question=self.question,
        )


def _suggestion_block(feedback) -> str:
    if not feedback:
        return ""
    return P.SUGGESTION_BLOCK.format(
        previous=feedback["previous"], suggestions=feedback["suggestions"] or "None"
    )


class BaseRefineTool:
    name = "base"

    def __init__(self, llm, n: int = 5, max_table_rows: int | None = None) -> None:
        self.llm = llm
        self.n = n
        self.max_table_rows = max_table_rows

    def generate(self, ctx: RefineContext, feedback=None) -> list[Candidate]:
        raise NotImplementedError

    def critic(self, ctx: RefineContext, cand: Candidate):
        raise NotImplementedError


@register_refine_tool("cot")
class CoTTool(BaseRefineTool):
    """Plain chain-of-thought regeneration (no external tool)."""

    def generate(self, ctx, feedback=None):
        prompt = P.INIT.format(
            fewshot=P.FEWSHOT_INIT[ctx.task],
            table=ctx.table.to_markdown(self.max_table_rows),
            **ctx.fmt,
        )
        sug = _suggestion_block(feedback)
        if sug:
            prompt = prompt.rstrip() + "\n" + sug + "\nRe-solve the problem taking the suggestions into account.\n"
        outs = self.llm.sample(prompt, self.n)
        return [
            Candidate(self.name, extract_answer(o), normalize(extract_answer(o), ctx.task), o)
            for o in outs
        ]

    def critic(self, ctx, cand):
        prompt = P.CRITIC_INIT.format(
            table=ctx.table.to_markdown(self.max_table_rows),
            reasoning=cand.trace,
            answer=cand.answer or "None",
            **ctx.fmt,
        )
        prompt = prompt.replace(
            "refinement with tools is REQUIRED, because LLMs are unreliable at arithmetic",
            "re-check every arithmetic step carefully",
        )
        out = self.llm.chat(prompt, temperature=0.0)
        return (*parse_critic(out), out)


@register_refine_tool("subtable")
class SubTableTool(BaseRefineTool):
    """Sub-table extraction for long / noisy tables."""

    def _extract(self, ctx, sug):
        prompt = P.SUBTABLE_SELECT.format(
            table=ctx.table.to_markdown(self.max_table_rows), suggestions=sug, **ctx.fmt
        )
        out = self.llm.chat(prompt, temperature=0.0)
        js = extract_json(out)
        cols = js.get("columns") if isinstance(js.get("columns"), list) else None
        rows = js.get("rows") if isinstance(js.get("rows"), list) else None
        sub = ctx.table.sub_table(cols, rows)
        if rows:
            ids = sorted(
                {
                    int(i)
                    for i in rows
                    if str(i).lstrip("-").isdigit() and 0 <= int(i) < len(ctx.table)
                }
            )
        else:
            ids = list(range(len(sub)))
        return sub, ids, js

    def _render(self, sub, ids):
        md = sub.to_markdown().splitlines()
        body_start = next(i for i, line in enumerate(md) if line.startswith("|---")) + 1
        for k, i in enumerate(ids[: len(md) - body_start]):
            line = md[body_start + k]
            md[body_start + k] = f"| {i} |" + line.split("|", 2)[2]
        return "\n".join(md)

    def generate(self, ctx, feedback=None):
        sug = _suggestion_block(feedback)
        sub, ids, _ = self._extract(ctx, sug)
        sub_md = self._render(sub, ids) if len(ids) == len(sub) else sub.to_markdown()
        prompt = P.SUBTABLE_ANSWER.format(
            fewshot=P.FEWSHOT_INIT[ctx.task], table=sub_md, suggestions=sug, **ctx.fmt
        )
        outs = self.llm.sample(prompt, self.n)
        return [
            Candidate(self.name, a := extract_answer(o), normalize(a, ctx.task), o, {"subtable": sub_md})
            for o in outs
        ]

    def critic(self, ctx, cand):
        prompt = P.CRITIC_SUBTABLE.format(
            table=ctx.table.to_markdown(self.max_table_rows),
            subtable=cand.extra.get("subtable", ""),
            reasoning=cand.trace,
            answer=cand.answer,
            **ctx.fmt,
        )
        out = self.llm.chat(prompt, temperature=0.0)
        return (*parse_critic(out), out)


_RUNNER = textwrap.dedent("""
    import sys, json, math, re, datetime
    import pandas as pd
    import numpy as np
    _p = json.loads(sys.stdin.read())
    df = pd.DataFrame(_p["rows"], columns=_p["header"])
    answer = None
    exec(compile(_p["code"], "<llm_code>", "exec"))
    def _c(x):
        if isinstance(x, (np.generic,)): x = x.item()
        if isinstance(x, (pd.Series, pd.Index, np.ndarray)): x = list(x)
        if isinstance(x, (list, tuple, set)): return [_c(i) for i in x]
        if isinstance(x, pd.DataFrame): return x.to_string()
        return x if isinstance(x, (str, int, float, bool)) or x is None else str(x)
    print("__ANSWER__" + json.dumps(_c(answer), ensure_ascii=False, default=str))
""")


def run_python(code: str, table, timeout: int = 15):
    """Execute LLM code in a separate process. Returns (ok, answer, log)."""

    payload = json.dumps({"code": code, **table.to_payload()}, ensure_ascii=False)
    try:
        process = subprocess.run(
            [sys.executable, "-c", _RUNNER],
            input=payload, text=True, capture_output=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, None, f"TimeoutError: code execution exceeded {timeout}s"
    out = process.stdout
    if "__ANSWER__" in out:
        answer = json.loads(out.split("__ANSWER__")[-1].strip())
        log = out.split("__ANSWER__")[0][-1500:]
        return answer is not None, answer, log or "OK"
    return False, None, (process.stderr or out)[-1500:]


@register_refine_tool("python")
class PythonTool(BaseRefineTool):
    """Python code generation + execution (numerical reasoning / look-up)."""

    def generate(self, ctx, feedback=None):
        table = ctx.table
        answer_type = (
            " `answer` must be a Python bool (True if the statement is entailed, else False)."
            if ctx.task == "tfv"
            else " `answer` should be a string, a number, or a list (for multiple answers)."
        )
        prompt = P.PYTHON_GEN.format(
            context=f"Context: {table.context}\n" if table.context else "",
            columns=table.to_df().columns.tolist(),
            n_rows=len(table),
            table=table.to_markdown(self.max_table_rows or 30),
            suggestions=_suggestion_block(feedback),
            answer_type=answer_type,
            **ctx.fmt,
        )
        outs = self.llm.sample(prompt, self.n)
        candidates = []
        for o in outs:
            code = extract_code(o)
            ok, ans, log = run_python(code, table)
            if isinstance(ans, list) and ctx.task == "tqa":
                raw = " | ".join(str(x) for x in ans)
            else:
                raw = "" if ans is None else str(ans)
            candidates.append(
                Candidate(
                    self.name, raw, normalize(ans, ctx.task) if ok else "", code,
                    {"exec_ok": ok, "exec_log": log},
                )
            )
        return candidates

    def critic(self, ctx, cand):
        result = (
            cand.answer
            if cand.extra.get("exec_ok")
            else f"None\nError log:\n{cand.extra.get('exec_log')}"
        )
        prompt = P.CRITIC_PYTHON.format(
            table=ctx.table.to_markdown(self.max_table_rows),
            code=cand.trace,
            exec_result=result,
            answer=cand.answer or "None",
            **ctx.fmt,
        )
        out = self.llm.chat(prompt, temperature=0.0)
        need, sug = parse_critic(out)
        if not cand.extra.get("exec_ok"):
            need = True
        return need, sug, out


__all__ = [
    "BaseRefineTool",
    "Candidate",
    "REFINE_TOOL_LIBRARY",
    "RefineContext",
    "register_refine_tool",
    "run_python",
]
