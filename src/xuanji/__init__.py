"""XuanJi (璇玑): unified question answering over tables and databases.

Named after Su Hui's 璇玑图 — an 841-character palindrome grid from which
thousands of poems can be read — XuanJi reads answers out of tables: one
grid, many questions.

One pipeline covers the whole chain: optional ambiguity detection and
clarification (CLEAR-style), an agentic ReAct solver with self-consistency
for long tables and databases (TALON/RAISR families), and an iterative
self-criticism refiner with a pluggable tool library (TableCritic family).

    import xuanji

    engine = xuanji.XuanJi()            # endpoint from XUANJI_* / OPENAI_* env
    result = engine.ask("Which venue had the most wins?", table=table)
    print(result.answer)

For offline development and tests, inject ``xuanji.FakeLLM``.
"""

from xuanji.disambiguate import DisambiguationResult, Disambiguator
from xuanji.eval import denotation_accuracy, load_dataset
from xuanji.llm import LLM, ChatClient, FakeLLM
from xuanji.pipeline import XuanJi, XuanJiResult, ask
from xuanji.registry import ToolContext, ToolRegistry, ToolSpec, register_tool
from xuanji.table import Table
from xuanji.tools.db import Database

__version__ = "0.2.0"

__all__ = [
    "ChatClient",
    "Database",
    "DisambiguationResult",
    "Disambiguator",
    "FakeLLM",
    "LLM",
    "Table",
    "ToolContext",
    "ToolRegistry",
    "ToolSpec",
    "XuanJi",
    "XuanJiResult",
    "__version__",
    "ask",
    "denotation_accuracy",
    "load_dataset",
    "register_tool",
]
