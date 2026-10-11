"""DataFrame and action-parsing utilities for the solver loop.

Adapted from the TALON repository (Apache-2.0, Copyright 2024 The Google
Research Authors).  Action parsing is generalized: the set of legal tool
names is supplied by the registry instead of being hard-coded.
"""

from __future__ import annotations

import ast
import io
import re

import pandas as pd


def infer_dtype(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt numeric, then datetime conversion per column; failures keep strings."""

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="ignore")
            if df[col].dtype == "object":
                df[col] = pd.to_datetime(df[col], errors="raise")
        except Exception:  # preserve original lenient semantics
            pass
    return df


def unique_columns(columns) -> list[str]:
    """Normalize line breaks and suffix duplicates without colliding with existing names."""

    names = [str(col).replace("\n", " ").replace("\\n", " ") for col in columns]
    reserved, used, result = set(names), set(), []
    for name in names:
        candidate, suffix = name, 2
        if candidate in used:
            while f"{name}_{suffix}" in reserved or f"{name}_{suffix}" in used:
                suffix += 1
            candidate = f"{name}_{suffix}"
        used.add(candidate)
        result.append(candidate)
    return result


def get_df_info(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.info(verbose=True, buf=buffer)
    return buffer.getvalue()


def extract_value_from_string(call: str) -> str:
    node = ast.parse(call, mode="eval").body
    if not isinstance(node, ast.Call) or len(node.args) != 1 or node.keywords:
        raise ValueError("Tool requires one literal argument")
    value = ast.literal_eval(node.args[0])
    if not isinstance(value, (str, int, float)):
        raise ValueError("Tool argument must be a string or number")
    return str(value)


def is_python_action(action: str) -> bool:
    return bool(re.match(r"^(?:def\(\)\s*)?```(?:python)?\s", action, re.I))


def extract_action(response: str, tool_names: set[str], no_arg_tools: set[str] | None = None) -> list[str]:
    """Return same-type tool calls, or one fenced Python action; never execute arguments.

    ``tool_names`` is the registry's legal action space; ``no_arg_tools``
    lists tools callable without arguments (e.g. get_column_meaning).
    """

    no_arg_tools = no_arg_tools or set()
    match = re.search(r"(?:^|\n)\s*Action:\s*(.*)", response, re.S | re.I)
    if not match:
        return []
    action = match.group(1).strip()
    if is_python_action(action):
        return [action] if re.search(r"```(?:python)?\s.*?```", action, re.S | re.I) else []
    # Treat repeated Action lines as one requested batch; never silently drop later calls.
    action = re.sub(r"(?im)^\s*Action:\s*", ", ", action)
    try:
        node = ast.parse(f"({action})", mode="eval").body
        calls = node.elts if isinstance(node, ast.Tuple) else [node]
        names, result = [], []
        for call in calls:
            if (
                not isinstance(call, ast.Call)
                or not isinstance(call.func, ast.Name)
                or call.func.id not in tool_names
                or call.keywords
            ):
                return []
            name = call.func.id
            if name in no_arg_tools:
                if call.args:
                    return []
            else:
                extract_value_from_string(ast.unparse(call))
            names.append(name)
            result.append(ast.unparse(call))
        return result if len(set(names)) == 1 else []
    except (ValueError, TypeError, SyntaxError):
        return []


FINAL_ANSWER = re.compile(r"(?:^|\n)\s*Final\s+Answer:\s*([^\n]*)", re.I)


def is_terminal(text: str) -> bool:
    return bool(FINAL_ANSWER.search(text or ""))


def final_answer(text: str) -> str:
    match = FINAL_ANSWER.search(text or "")
    return match.group(1).strip() if match else ""
