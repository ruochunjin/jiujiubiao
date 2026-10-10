"""XuanJi (璇玑): unified question answering over tables and databases.

Named after Su Hui's 璇玑图 — an 841-character palindrome grid from which
thousands of poems can be read — XuanJi reads answers out of tables: one
grid, many questions.

One pipeline covers the whole chain: optional ambiguity detection and
clarification (CLEAR-style), an agentic ReAct solver with self-consistency
for long tables and databases (TALON/RAISR families), and an iterative
self-criticism refiner with a pluggable tool library (TableCritic family).

    import xuanjitu

    engine = xuanjitu.XuanJi()            # endpoint from XUANJI_* / OPENAI_* env
    result = engine.ask("Which venue had the most wins?", table=table)
    print(result.answer)

For offline development and tests, inject ``xuanjitu.FakeLLM``.
"""

from xuanjitu.disambiguate import DisambiguationResult, Disambiguator
from xuanjitu.eval import denotation_accuracy, load_dataset
from xuanjitu.llm import LLM, AnthropicClient, ChatClient, FakeLLM
from xuanjitu.pipeline import XuanJi, XuanJiResult, ask
from xuanjitu.registry import ToolContext, ToolRegistry, ToolSpec, register_tool
from xuanjitu.table import Table
from xuanjitu.tools.db import Database

__version__ = "0.3.0"

__all__ = [
    "AnthropicClient",
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
