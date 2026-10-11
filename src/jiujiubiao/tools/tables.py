"""Built-in table-exploration tools (TALON algorithm family).

Pure tools (``get_value`` / ``get_row`` / ``find_columns_containing_value_fuzzy``)
need no model.  ``get_column_meaning`` and ``find_column_format`` call the
LLM through the tool context, and ``find_column_format`` can rewrite
``ctx.memory["df"]`` (e.g. converting "79,483" text columns to numbers).

Fuzzy matching uses stdlib difflib by default; install ``rapidfuzz`` for a
faster, token-set-aware matcher (auto-detected).
"""

from __future__ import annotations

import difflib
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd

from jiujiubiao.execution import parse_code_from_string, python_repl_ast
from jiujiubiao.prompts.solve import DESCRIPTION_PROMPT, FORMAT_PROMPT
from jiujiubiao.registry import ToolContext, register_tool

try:  # optional speedup
    from rapidfuzz import fuzz as _rf_fuzz

    def _similarity(a: str, b: str) -> float:
        return float(_rf_fuzz.token_set_ratio(a, b))

except ImportError:

    def _similarity(a: str, b: str) -> float:
        """Token-set-ratio approximation on stdlib difflib (0–100)."""

        tokens_a, tokens_b = set(a.split()), set(b.split())
        shared = sorted(tokens_a & tokens_b)
        base = difflib.SequenceMatcher(None, a, b).ratio()
        if shared:
            joined = " ".join(shared)
            best = max(
                base,
                difflib.SequenceMatcher(None, joined, " ".join(sorted(tokens_a))).ratio(),
                difflib.SequenceMatcher(None, joined, " ".join(sorted(tokens_b))).ratio(),
            )
            return 100.0 * best
        return 100.0 * base


EXEC_GLOBALS = {"pd": pd, "np": np, "re": re, "datetime": datetime}


def _df_or_hint(ctx: ToolContext):
    """(df, hint): the working dataframe, or a steering message when absent.

    Database mode (ctx.df None) has no pandas frame — table tools must not
    crash on it (CLAMBSQL smoke, 2026-10-10); steer the agent to execute_sql.
    """

    df = ctx.memory.get("df", ctx.df)
    if df is None:
        if getattr(ctx, "db", None) is not None:
            return None, (
                "This tool operates on pandas tables and cannot run on a "
                "database; use execute_sql() with SQL against the database instead."
            )
        return None, "No table is loaded for this tool."
    return df, None


@register_tool(takes_arg=True)
def get_value(ctx: ToolContext, value: str) -> str:
    """get_value(value) — Search for a specific value across all columns; tells you which columns contain it and at which row indices."""
    df, hint = _df_or_hint(ctx)
    if hint:
        return hint
    search_value = str(value)
    if search_value.casefold() in [str(col).casefold() for col in df.columns]:
        return "The value is a column name; use get_column_meaning() or find_column_format()."
    matches = []
    for col in df.columns:
        values = df[col].astype(str)
        for cell in values.unique():
            if not search_value:
                match = not cell.strip()
            else:
                try:
                    number = Decimal(cell)
                except InvalidOperation:
                    match = search_value.casefold() in cell.casefold()
                else:
                    try:
                        match = number == Decimal(search_value)
                    except InvalidOperation:
                        match = False
            if match:
                indices = df.index[values == cell].tolist()
                suffix = f" (first 5 of {len(indices)})" if len(indices) > 5 else ""
                matches.append(
                    f"[Value]: {cell} -[Column Name]: {col} -[row_indices]: {indices[:5]}{suffix}"
                )
    if not matches:
        return f"No columns contain '{search_value}'. Try find_columns_containing_value_fuzzy()."
    return "The columns below contain the value:\n" + "\n".join(matches)


@register_tool(takes_arg=True)
def get_row(ctx: ToolContext, value: str) -> str:
    """get_row(row_index) — Retrieve one row by integer index, e.g. to understand relationships between columns in that row."""
    df, hint = _df_or_hint(ctx)
    if hint:
        return hint
    try:
        index = int(str(value))
    except ValueError:
        return "Row index must be an integer."
    if index < 0 or index >= len(df):
        return "Row index out of range."
    return f"Row {index} content: {df.iloc[index].to_dict()}"


@register_tool(takes_arg=True)
def find_columns_containing_value_fuzzy(ctx: ToolContext, value: str, threshold: float = 70, top_k: int = 5) -> str:
    """find_columns_containing_value_fuzzy(value) — Fuzzy-search the whole table for a value that may be spelled differently (abbreviations, reordering, typos)."""
    df, hint = _df_or_hint(ctx)
    if hint:
        return hint
    search_str = str(value).lower()
    results = []

    def process_column(col_name):
        col_results = []
        for idx, cell in df[col_name].astype(str).str.lower().items():
            if cell == "nan":
                continue
            score = _similarity(search_str, cell)
            if score >= threshold:
                col_results.append(
                    {"row_index": idx, "column": col_name, "value": df.at[idx, col_name], "score": score}
                )
        return col_results

    with ThreadPoolExecutor(max_workers=4) as executor:
        for col_results in executor.map(process_column, df.columns.tolist()):
            results.extend(col_results)

    if not results:
        return f"No matches found for '{value}' in the table."
    results.sort(key=lambda r: -r["score"])
    lines = [f"The columns below may contain relevant values for '{value}':"]
    for row in results[:top_k]:
        lines.append(f"- Column [{row['column']}] contains: '{row['value']}' (row {row['row_index']}, score {row['score']:.0f})")
    return "\n".join(lines)


@register_tool(takes_arg=False, needs_llm=True)
def get_column_meaning(ctx: ToolContext, _arg: str | None = None) -> str:
    """get_column_meaning() — Ask the LLM which columns are most relevant to the question, with per-column meanings and value statistics."""
    df = ctx.memory.get("df", ctx.df)
    if df is None:
        # Database mode: present the schema instead of a pandas preview.
        if getattr(ctx, "db", None) is not None:
            schema = ctx.db.schema_text(sample_values=2)
            prompt = DESCRIPTION_PROMPT.format(
                preview_data=schema, question=ctx.question, thought=ctx.thought
            )
            return ctx.llm.chat(prompt, temperature=0.8, stop=["Observation:"], max_tokens=512)
        return "No table is loaded for this tool."
    preview = df.head(5).to_string(index=False) + "\n" + df.tail(1).to_string(index=False)
    prompt = DESCRIPTION_PROMPT.format(
        preview_data=preview, question=ctx.question, thought=ctx.thought
    )
    response = ctx.llm.chat(prompt, temperature=0.8, stop=["Observation:"], max_tokens=512)
    for col in df.columns:
        if col in response:
            values = df[col].drop_duplicates()
            response += (
                f"\nColumn '{col}': {len(values)} unique values; "
                f"first 10: {values.head(10).tolist()}; dtype: {df[col].dtype}."
            )
    return response


@register_tool(takes_arg=True, needs_llm=True)
def find_column_format(ctx: ToolContext, value: str, max_repairs: int = 3) -> str:
    """find_column_format(column) — Inspect and (when needed) repair the format of a column via LLM-written pandas code, e.g. convert '79,483' text to numbers."""
    df, hint = _df_or_hint(ctx)
    if hint:
        return hint
    column_name = str(value)
    if column_name not in df.columns:
        return f"Column '{column_name}' not found in the DataFrame. Available: {list(df.columns)}"
    column_dtype = df[column_name].dtype
    sampled = df[column_name].drop_duplicates().head(20).tolist()
    prompt = FORMAT_PROMPT.format(
        column_name=column_name, column_dtype=column_dtype, sampled_values=sampled, thought=ctx.thought
    )
    response = ctx.llm.chat(prompt, temperature=0.8, stop=["Observation:"], max_tokens=512)
    if "[[No]]" in response:
        return f"Column '{column_name}' has dtype {column_dtype}; examples: {sampled}."
    match = re.search(r"```(?:python)?\s*(.*?)```", response, re.S)
    if not match:
        return "Error: no Python code provided for column formatting."
    code = match.group(1)
    for attempt in range(max_repairs + 1):
        observation, updated, success = python_repl_ast(
            parse_code_from_string(code) if code else "",
            custom_locals={"df": df},
            custom_globals=EXEC_GLOBALS,
            memory=ctx.memory,
        )
        if success:
            ctx.memory.update(updated)
            new_df = ctx.memory["df"]
            examples = new_df[column_name].head(5).tolist() if column_name in new_df else []
            return f"Column '{column_name}' processed; examples: {examples}. {observation}"
        if attempt >= max_repairs or ctx.critic is None:
            return f"Error processing column '{column_name}': {observation}"
        repaired = ctx.critic.critic_python_tool(code, observation, sampled, ctx.thought, column_dtype)
        if not repaired:
            return f"Error processing column '{column_name}': {observation}"
        code = repaired
    return f"Error processing column '{column_name}'."
