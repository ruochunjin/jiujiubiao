"""Critic services for the solver loop (TALON family, Apache-2.0 adapted).

Two failure modes are repaired here:

- **loop resolution** — the solver repeats the same action; the critic is
  asked to rewrite the next thought/action from the trajectory;
- **code repair** — generated Python failed; the critic rewrites it using
  the error message (and, for column formatting, sampled values).
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class Critic:
    """LLM-backed repair callbacks; all calls go through the injected LLM."""

    def __init__(self, llm: Any, temperature: float = 0.8, max_tokens: int = 512) -> None:
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens

    def query(self, prompt: str) -> str:
        return self.llm.chat(
            prompt,
            temperature=self.temperature,
            stop=["Observation:"],
            max_tokens=self.max_tokens,
        )

    def critic_python_tool(self, python_code, error, sampled_values, thought, column_dtype):
        """Repair code that failed while formatting a column."""

        prompt = f"""You have executed a python code when processing the format of the column,
        the thought of process the column is:
        {thought},
        the sample values of the column are {sampled_values},
        and the code is:
        {python_code},
        but it has an error, here is an error message:
        {error}

        Analyze the error message to:

        1. Diagnose the root cause of the failure .
        2. Validate the code logic against the provided DataFrame (df) based on the sample values.
        3. Formulate a corrected code, you can generate different code to process the column based on the thought and the sample values to avoid such an error.
        4. If Sample data of the table, you should modify the code to avoid sampling data.

        Critical Constraints:
        - The table is already available as a pandas DataFrame (df)
        - Never sample the table data during analysis.

        please think step by step and output the correct code with the format of ```python
        {python_code}```.
        """
        response = self.query(prompt)
        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        return code_match.group(1) if code_match else None

    def critic_circle(self, trajectory: str, action) -> tuple[str, str, str]:
        """Rewrite the next step after the solver repeats itself (loop resolution)."""

        prompt = f"""As an AI assistant analyzing tabular data, I've noticed you've repeatedly called the same data processing tool multiple times. Based on previous observations, please:

            1. Re-evaluate your current approach
            2. Generate a new 'thought' process that considers:
            - Efficiency improvements
            - Avoiding redundant operations including all the history actions
            - Leveraging previously obtained results from the trajectory
            3. Provide a revised 'action' plan that:
            - avoid duplicate tool calls
            - generate a tool or final answer based on the trajectory


            Format your response with clear "Thought:" and "Action:" sections, ensuring the solution is both technically sound and resource-efficient.
             Output Format:
            Thought: Only print about the thinking process about what to do next .
            Action: Directly output the tool you choosed to use, do not output other inforamtion.

            Here is the trajectory: {trajectory}
            Here is the repeated action: {action}

        """
        text = self.query(prompt)
        thought = text.split("Action:")[0].replace("Thought:", "").strip()
        parts = text.split("Action:")
        action = "Action:" + "Action:".join(parts[1:]).strip()
        return text, thought, action

    def critic_python(self, trajectory: str, error: str) -> str | None:
        """Repair solver-generated Python using the trajectory and error."""

        prompt = f"""
        You have to read the table and the query, and all the thinking process, to identity the root cause of the error.
        Analyze the execution trace and error message to:

        1. Diagnose the root cause of the failure based on the error message and the trajectory.
        2. Validate the code logic against the provided DataFrame (df) based on the observations.
        3. Formulate a corrected code.
        4. If Sample data of the table, you should modify the code to avoid sampling data.

        Critical Constraints:
        - The table is already available as a pandas DataFrame (df)
        - Never sample the table data during analysis.

        Please Note: You should directly output the revised python code with the format of ```python\n``` to avoid such an error,no other additional text.


        Here is the thinking process:
        {trajectory}
        Here is the error message:
        {error}

        """
        response = self.query(prompt)
        match = re.search(r"```(?:python)?\s*(.*?)\s*```", response, re.DOTALL)
        return match.group(1) if match else None


__all__ = ["Critic"]
