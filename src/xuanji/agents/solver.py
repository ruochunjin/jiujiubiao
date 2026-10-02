"""ReAct solver with self-consistency (TALON family, generalized).

The loop — sample ``sc`` candidate thoughts, majority-vote the next action,
execute it through the tool registry, append the observation — is preserved
from TALON, but the action space is no longer hard-coded: any tool in the
registry can be dispatched, and its signature/description are rendered into
the prompt automatically.

Loop resolution (repeated action → critic rewrite) and code repair
(failed Python → critic repair, up to 4 attempts) come from the injected
:class:`~xuanji.agents.critic.Critic`.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from xuanji.agents.critic import Critic
from xuanji.df_utils import (
    extract_action,
    extract_value_from_string,
    final_answer,
    get_df_info,
    is_python_action,
    is_terminal,
)
from xuanji.execution import parse_code_from_string, python_repl_ast
from xuanji.prompts.solve import TASK_TEMPLATES
from xuanji.registry import DEFAULT_REGISTRY, ToolContext, ToolRegistry

EXEC_GLOBALS = {"pd": pd, "re": re}
try:
    import numpy as np

    EXEC_GLOBALS["np"] = np
except ImportError:  # pragma: no cover
    pass


@dataclass(slots=True)
class SolveResult:
    """One solver run."""

    answer: str
    steps: int
    trajectory: str
    status: str  # "completed" | "incomplete"
    tool_counts: dict[str, int] = field(default_factory=dict)


def route_check(action_stack: list[str], tools: list[str]) -> bool:
    """Record a whole action atomically, only if none of its calls have occurred."""

    if any(tool in action_stack for tool in tools):
        return False
    action_stack.extend(tools)
    return True


def find_most_frequent_tool_index(tools_list: list[list[str]]) -> int | None:
    counter = Counter(tool for tools in tools_list for tool in tools)
    if not counter:
        return None
    maximum = max(counter.values())
    candidates = [
        i for i, tools in enumerate(tools_list) if any(counter[tool] == maximum for tool in tools)
    ]
    return min(candidates, key=lambda i: (-len(tools_list[i]), i))


class Solver:
    """The ReAct + self-consistency table/database solver."""

    def __init__(
        self,
        llm: Any,
        registry: ToolRegistry = DEFAULT_REGISTRY,
        *,
        sc: int = 5,
        max_depth: int = 10,
        temperature: float = 0.8,
        max_tokens: int = 1280,
        use_critic: bool = True,
        verbose: bool = False,
    ) -> None:
        if sc < 1 or max_depth < 1 or max_tokens < 1:
            raise ValueError("sc, max_depth and max_tokens must be positive")
        self.llm = llm
        self.registry = registry
        self.sc = sc
        self.max_depth = max_depth
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.critic = Critic(llm) if use_critic else None

    # ------------------------------------------------------------------ prompt
    def build_prompt(
        self, task: str, question: str, df_info: str, caption: str = "", evidence: str = ""
    ) -> str:
        template = TASK_TEMPLATES.get(task, TASK_TEMPLATES["wtq"])
        prompt = template.format(
            query=question, table=df_info, table_caption=caption, evidence=evidence
        )
        extra = [
            f"- `{spec.signature}`: {spec.description}"
            for spec in self.registry
            if spec.name not in _BUILTIN_TOOL_NAMES
        ]
        if extra:
            prompt += "\nAdditional available tools:\n" + "\n".join(extra) + "\n"
        return prompt

    # ------------------------------------------------------------------ steps
    def _vote_step(self, prompt: str, memory: dict) -> str:
        responses = self.llm.sample(
            prompt,
            self.sc,
            temperature=self.temperature,
            stop=["Observation:"],
            max_tokens=self.max_tokens,
        )
        votes = []
        for response in responses:
            if not response or not response.strip():
                votes.append([])
                continue
            if is_terminal(response):
                votes.append(["answer:" + final_answer(response)])
                continue
            actions = extract_action(response, self.registry.names(), self.registry.no_arg_names())
            if actions and is_python_action(actions[0]):
                code = parse_code_from_string(actions[0])
                observation, _, success = python_repl_ast(
                    code, custom_locals={"df": memory.get("df")},
                    custom_globals=EXEC_GLOBALS, memory=memory,
                )
                actions = ["python:" + observation] if success else []
            votes.append(actions)
        selected = find_most_frequent_tool_index(votes)
        if selected is not None:
            return responses[selected]
        return next((text for text in responses if text and text.strip()), "")

    def _execute_actions(
        self, actions: list[str], action_text: str, memory: dict, trajectory: str, counts: dict
    ) -> tuple[str, dict]:
        if not actions:
            return "Error: no valid Action provided. Use one tool type per action.", memory
        if is_python_action(actions[0]):
            counts["def"] += 1
            code = parse_code_from_string(action_text)
            for attempt in range(4):
                observation, memory, success = python_repl_ast(
                    code, custom_locals={"df": memory.get("df")},
                    custom_globals=EXEC_GLOBALS, memory=memory,
                )
                if success:
                    return (
                        observation.strip() or "Code executed successfully; nothing was printed.",
                        memory,
                    )
                if attempt < 3 and self.critic is not None:
                    code = self.critic.critic_python(trajectory, observation) or ""
                    counts["critic_python"] += 1
            return f"Error: Python repair failed: {observation}", memory
        observations = []
        for action in actions:
            name = action.split("(", 1)[0]
            counts[name] += 1
            spec = self.registry.get(name)
            arg = None if not spec.takes_arg else extract_value_from_string(action)
            ctx = ToolContext(
                df=memory.get("df"), llm=self.llm, question=self._question,
                memory=memory, thought=self._thought, db=self._db, critic=self.critic,
            )
            observations.append(spec.run(ctx, arg))
            memory.update(ctx.memory)
        return "\n".join(observations), memory

    # ------------------------------------------------------------------ main
    def solve(
        self,
        df: pd.DataFrame | None,
        question: str,
        *,
        task: str = "wtq",
        caption: str = "",
        evidence: str = "",
        db: Any = None,
    ) -> SolveResult:
        """Solve one question over a DataFrame (or an attached Database)."""

        self._question, self._db = question, db
        self._thought = ""
        memory: dict = {"df": df} if df is not None else {}
        prompt = self.build_prompt(
            task, question, get_df_info(df) if df is not None else db.schema_text(), caption, evidence
        )
        counts: dict[str, int] = dict.fromkeys(
            list(self.registry.names()) + ["def", "critic_circle", "critic_python"], 0
        )
        action_stack: list[str] = []
        solution = ""
        for step in range(1, self.max_depth + 1):
            text = self._vote_step(prompt + solution + "Thought: ", memory).strip()
            actions = extract_action(text, self.registry.names(), self.registry.no_arg_names())
            while (
                self.critic is not None
                and not is_terminal(text)
                and actions
                and not route_check(action_stack, actions)
            ):
                if counts["critic_circle"] >= 5:
                    break
                text, _, _ = self.critic.critic_circle(prompt + solution, actions)
                counts["critic_circle"] += 1
                actions = extract_action(text, self.registry.names(), self.registry.no_arg_names())
            if self.verbose:
                print(text)
            if is_terminal(text):
                solution += text + "\n"
                return SolveResult(final_answer(text), step, solution, "completed", counts)
            self._thought = re.split(r"Action:", text, maxsplit=1, flags=re.I)[0].strip()
            observation, memory = self._execute_actions(
                actions, actions[0] if actions else "", memory, prompt + solution + text, counts
            )
            solution += text + f"\nObservation: {observation}\n"
            if self.verbose:
                print(f"Observation: {observation}")
        # An unfinished thought/action must never be reported as the final answer.
        return SolveResult("", self.max_depth, solution, "incomplete", counts)


_BUILTIN_TOOL_NAMES = {
    "get_column_meaning",
    "find_columns_containing_value_fuzzy",
    "find_column_format",
    "get_value",
    "get_row",
}


__all__ = ["SolveResult", "Solver"]
