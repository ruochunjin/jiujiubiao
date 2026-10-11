"""JiuJiuBiao (璇玑): unified question answering over tables and databases.

Named after Su Hui's 璇玑图 — an 841-character palindrome grid from which
thousands of poems can be read — JiuJiuBiao reads answers out of tables: one
grid, many questions.

One pipeline covers the whole chain: optional ambiguity detection and
clarification (CLEAR-style), an agentic ReAct solver with self-consistency
for long tables and databases (TALON/RAISR families), and an iterative
self-criticism refiner with a pluggable tool library (TableCritic family).

    import jiujiubiao

    engine = jiujiubiao.JiuJiuBiao()            # endpoint from JIUJIUBIAO_* / OPENAI_* env
    result = engine.ask("Which venue had the most wins?", table=table)
    print(result.answer)

For offline development and tests, inject ``jiujiubiao.FakeLLM``.
"""

from jiujiubiao.disambiguate import DisambiguationResult, Disambiguator
from jiujiubiao.eval import denotation_accuracy, load_dataset
from jiujiubiao.llm import LLM, AnthropicClient, ChatClient, FakeLLM
from jiujiubiao.pipeline import JiuJiuBiao, JiuJiuBiaoResult, ask
from jiujiubiao.registry import ToolContext, ToolRegistry, ToolSpec, register_tool
from jiujiubiao.table import Table
from jiujiubiao.tools.db import Database

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
    "JiuJiuBiao",
    "JiuJiuBiaoResult",
    "__version__",
    "ask",
    "denotation_accuracy",
    "load_dataset",
    "register_tool",
]
