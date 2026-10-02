"""CORES-style composite process reward for code-oriented table reasoning.

Follows the paper's formulation (Algorithm 2): a response must contain
``<nl_reasoning>``, ``<python_reasoning>`` and ``<sql>`` sections (format
gate); the reward then combines

- **schema grounding**: weighted F1 of entities mentioned in each section
  against the gold SQL's entities (weights NL 0.5, Python 1.0, SQL 1.0);
- **structure**: skeleton/sequence similarity of Python and SQL to gold;
- **execution**: both snippets actually run against the database and match
  the gold result (0.25 × runs + 1.0 × matches, per language).

``R = R_f * (R_schema + R_struct + 2 * R_exec) / 9.5`` — normalized to [0, 1].

Everything is injected (schema dict, database path, executor); no global
state, no hardcoded paths.  stdlib + pandas only.
"""

from __future__ import annotations

import ast
import difflib
import multiprocessing
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# ----------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------


def parse_sections(response: str) -> tuple[str, str, str] | None:
    """Extract (nl, python, sql) sections; None when any is missing."""

    nl = re.findall(r"<nl_reasoning>(.*?)</nl_reasoning>", response, re.DOTALL)
    py = re.findall(r"<python_reasoning>(.*?)</python_reasoning>", response, re.DOTALL)
    sql = re.findall(r"<sql>(.*?)</sql>", response, re.DOTALL)
    if not (nl and py and sql):
        return None
    py_blocks = re.findall(r"```python(.*?)```", py[-1], re.DOTALL)
    sql_blocks = re.findall(r"```sql(.*?)```", sql[-1], re.DOTALL)
    if not (py_blocks and sql_blocks):
        return None
    sql_code = " ".join(sql_blocks[-1].replace("\n", " ").split())
    return nl[-1], py_blocks[-1].strip("\n"), sql_code


# ----------------------------------------------------------------------
# schema grounding
# ----------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "and", "or", "to", "is", "are",
    "was", "were", "by", "with", "as", "at", "from", "that", "this", "which",
}


def _entities_in_text(text: str, schema: Mapping[str, Sequence[str]]) -> set[str]:
    """Schema entities (tables/columns) mentioned in free text or code."""

    found: set[str] = set()
    lowered = text.lower()
    for table, columns in schema.items():
        if table.lower() in lowered:
            found.add(table)
        for column in columns:
            if re.search(rf"(?<![\w]){re.escape(column.lower())}(?![\w])", lowered):
                found.add(f"{table}.{column}")
    return found


def _entities_in_sql(sql: str, schema: Mapping[str, Sequence[str]]) -> set[str]:
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", sql))
    found: set[str] = set()
    for table, columns in schema.items():
        if table in tokens or table.lower() in {t.lower() for t in tokens}:
            found.add(table)
        for column in columns:
            if column in tokens:
                found.add(f"{table}.{column}")
    return found


def _f1(pred: set[str], gold: set[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    precision, recall = tp / len(pred), tp / len(gold)
    return 0.0 if tp == 0 else 2 * precision * recall / (precision + recall)


def schema_reward(
    nl: str, python_code: str, sql: str, gold_sql: str, schema: Mapping[str, Sequence[str]]
) -> float:
    """Weighted entity F1: NL 0.5 + Python 1.0 + SQL 1.0 (max 2.5)."""

    gold = _entities_in_sql(gold_sql, schema)
    return (
        0.5 * _f1(_entities_in_text(nl, schema), gold)
        + _f1(_entities_in_text(python_code, schema), gold)
        + _f1(_entities_in_sql(sql, schema), gold)
    )


# ----------------------------------------------------------------------
# structure
# ----------------------------------------------------------------------


class _Skeletonizer(ast.NodeTransformer):
    """Replace literals/names with type placeholders to get a code skeleton."""

    def visit_Constant(self, node):
        return ast.copy_location(
            ast.Constant(value=f"<{type(node.value).__name__.upper()}>"), node
        )

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id="VAR", ctx=node.ctx), node)


def _python_skeleton(code: str) -> str:
    try:
        tree = _Skeletonizer().visit(ast.parse(code))
        return ast.dump(tree, annotate_fields=False)
    except SyntaxError:
        return code


def _sql_skeleton(sql: str) -> str:
    sql = re.sub(r"'[^']*'|\b\d+(?:\.\d+)?\b", "<VAL>", sql)
    sql = re.sub(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", "<COL>", sql)
    return re.sub(r"\s+", " ", sql).strip().lower()


def structure_reward(python_code: str, gold_python: str, sql: str, gold_sql: str) -> float:
    """Sequence similarity of skeletons, Python + SQL (max 2.0)."""

    py = difflib.SequenceMatcher(None, _python_skeleton(python_code), _python_skeleton(gold_python)).ratio()
    sq = difflib.SequenceMatcher(None, _sql_skeleton(sql), _sql_skeleton(gold_sql)).ratio()
    return py + sq


# ----------------------------------------------------------------------
# execution
# ----------------------------------------------------------------------


def _exec_worker(db_path: str, kind: str, code: str, queue) -> None:
    try:
        if kind == "sql":
            conn = sqlite3.connect(db_path)
            try:
                rows = conn.execute(code).fetchall()
            finally:
                conn.close()
            queue.put(sorted([tuple(map(str, row)) for row in rows]))
        else:
            namespace: dict[str, Any] = {}
            import pandas as pd

            conn = sqlite3.connect(db_path)
            try:
                namespace["pd"] = pd
                namespace["sql"] = lambda q: pd.read_sql(q, conn)
                exec(compile(code, "<reward>", "exec"), namespace)
            finally:
                conn.close()
            result = namespace.get("result", namespace.get("answer"))
            if isinstance(result, pd.DataFrame):
                result = sorted(map(tuple, result.astype(str).values.tolist()))
            queue.put(result)
    except Exception:
        queue.put(None)


def _run_with_timeout(db_path: str, kind: str, code: str, timeout: float) -> Any:
    queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_exec_worker, args=(db_path, kind, code, queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return None
    try:
        return queue.get_nowait()
    except Exception:
        return None


def _results_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    if isinstance(a, list) and isinstance(b, list):
        return a == b
    return str(a) == str(b)


def execution_reward(
    python_code: str, sql: str, gold_sql: str, db_path: str, timeout: float = 10.0
) -> float:
    """0.25 × runs + 1.0 × matches-gold, for Python and SQL (max 2.5)."""

    gold = _run_with_timeout(db_path, "sql", gold_sql, timeout)
    py_result = _run_with_timeout(db_path, "python", python_code, timeout)
    sql_result = _run_with_timeout(db_path, "sql", sql, timeout)
    return (
        0.25 * (py_result is not None)
        + _results_equal(py_result, gold)
        + 0.25 * (sql_result is not None)
        + _results_equal(sql_result, gold)
    )


# ----------------------------------------------------------------------
# composite
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """The four reward dimensions and the normalized total."""

    format_ok: bool
    r_schema: float
    r_struct: float
    r_exec: float
    total: float  # in [0, 1]; 0 when the format gate fails


def compute_process_reward(
    response: str,
    gold_sql: str,
    gold_python: str,
    schema: Mapping[str, Sequence[str]],
    db_path: str,
    *,
    timeout: float = 10.0,
) -> RewardBreakdown:
    """Full composite reward for one response.

    ``schema``: ``{table_name: [column, ...]}``; ``db_path``: a SQLite file
    the snippets execute against.  Any dimension's failure degrades to 0 for
    that dimension without aborting the others.
    """

    sections = parse_sections(response)
    if sections is None:
        return RewardBreakdown(False, 0.0, 0.0, 0.0, 0.0)
    nl, python_code, sql = sections

    try:
        r_schema = schema_reward(nl, python_code, sql, gold_sql, schema)
    except Exception:
        r_schema = 0.0
    try:
        r_struct = structure_reward(python_code, gold_python, sql, gold_sql)
    except Exception:
        r_struct = 0.0
    try:
        r_exec = execution_reward(python_code, sql, gold_sql, db_path, timeout)
    except Exception:
        r_exec = 0.0

    total = (r_schema + r_struct + 2 * r_exec) / 9.5
    return RewardBreakdown(True, r_schema, r_struct, r_exec, total)


__all__ = ["RewardBreakdown", "compute_process_reward", "parse_sections"]
