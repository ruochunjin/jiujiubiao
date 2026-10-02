"""Tool registry: the pluggable action space for XuanJi agents.

Solver tools are plain functions wrapped in :class:`ToolSpec`; registering a
new tool makes it available to the solver's prompt builder and dispatcher::

    from xuanji.registry import ToolSpec, register_tool

    @register_tool
    def my_tool(ctx, arg):
        "count_rows() — return the number of rows in the table."
        return str(len(ctx.df))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator

import pandas as pd


@dataclass(slots=True)
class ToolContext:
    """Shared per-question state handed to every tool call."""

    df: pd.DataFrame | None
    llm: Any  # xuanji.llm.LLM
    question: str
    memory: dict  # execution state across def() calls; memory["df"] may evolve
    thought: str = ""
    db: Any = None  # xuanji.tools.db.Database when solving over a database
    critic: Any = None  # agents.critic.Critic, for self-repairing tools


ToolFn = Callable[[ToolContext, str | None], str]


@dataclass(slots=True)
class ToolSpec:
    """One registered tool: prompt-facing signature plus implementation."""

    name: str
    signature: str
    description: str
    fn: ToolFn
    takes_arg: bool = True
    needs_llm: bool = False

    def run(self, ctx: ToolContext, arg: str | None) -> str:
        return self.fn(ctx, arg)


class ToolRegistry:
    """Name → ToolSpec mapping with prompt-rendering support."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec, *, replace: bool = False) -> None:
        if spec.name in self._tools and not replace:
            raise ValueError(f"tool {spec.name!r} is already registered")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"unknown tool {name!r}; available: {sorted(self._tools)}") from None

    def names(self) -> set[str]:
        return set(self._tools)

    def no_arg_names(self) -> set[str]:
        return {spec.name for spec in self._tools.values() if not spec.takes_arg}

    def prompt_block(self) -> str:
        """Render the tool list for the solver system prompt."""

        return "\n".join(
            f"- `{spec.signature}`: {spec.description}"
            for spec in sorted(self._tools.values(), key=lambda s: s.name)
        )

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


DEFAULT_REGISTRY = ToolRegistry()


def register_tool(fn: ToolFn = None, *, name=None, signature=None, takes_arg=True, needs_llm=False):
    """Decorator: register ``fn`` with metadata from its docstring.

    The docstring's first line must be ``signature — description``.
    """

    def wrap(func: ToolFn) -> ToolFn:
        doc = (func.__doc__ or "").strip().splitlines()[0]
        sig, _, desc = doc.partition("—")
        if not desc:
            sig, _, desc = doc.partition("-")
        spec = ToolSpec(
            name=name or func.__name__,
            signature=signature or sig.strip() or f"{func.__name__}(value)",
            description=desc.strip() or "Custom tool.",
            fn=func,
            takes_arg=takes_arg,
            needs_llm=needs_llm,
        )
        DEFAULT_REGISTRY.register(spec)
        return func

    return wrap(fn) if fn is not None else wrap


__all__ = ["DEFAULT_REGISTRY", "ToolContext", "ToolRegistry", "ToolSpec", "register_tool"]
