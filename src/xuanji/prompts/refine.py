"""Prompt templates for TableCritic.

Every generator prompt ends with a line `Answer: <answer>` so answers can be parsed
uniformly; every critic prompt ends with `Need refinement: Yes/No` + `Suggestions:`.
"""

TASK_DESC = {
    "tqa": ("Answer the question based on the table. The answer should be short "
            "(an entity, a number, or a short phrase) and it ALWAYS exists in or can be derived "
            "from the table, so never answer 'unanswerable'/'N/A'; choose the most plausible interpretation. "
            "Copy entity names exactly as written in the table. If there are multiple answers, "
            "separate them with ' | '. "
            "Answer with the minimal span only: just the year/number/name — never append units "
            "('years', 'kg'), venues, ranks, or qualifiers unless the question asks for them. "
            "Boundary words matter: 'more than N' is strictly > N; 'at least N' is >= N. "
            "For multiple answers, verify EACH part against the table before including it. "
            "When counting entities (places, players, models), never count summary/total/"
            "remainder rows (e.g. 'Remainder of the municipality', 'Total') as entities."),
    "tfv": ("Verify whether the statement is entailed (True) or refuted (False) by the "
            "table. The answer must be exactly True or False."),
}

Q_NAME = {"tqa": "Question", "tfv": "Statement"}

# --------------------------------------------------------------------------
# few-shot examples (2 per task, as in the paper)
# --------------------------------------------------------------------------
FEWSHOT_INIT = {
    "tqa": """Example 1:
| row_id | Item | Unit Cost | Quantity |
|---|---|---|---|
| 0 | Laptops | 700 | 5 |
| 1 | Monitors | 150 | 10 |
| 2 | Notebooks | 10 | 20 |
Question: What is the total cost of all items?
Let's think step by step.
Laptops: 700 * 5 = 3500. Monitors: 150 * 10 = 1500. Notebooks: 10 * 20 = 200.
Total: 3500 + 1500 + 200 = 5200.
Answer: 5200

Example 2:
| row_id | Year | Venue | Position |
|---|---|---|---|
| 0 | 2001 | Edmonton | 3rd |
| 1 | 2003 | Paris | 1st |
| 2 | 2005 | Helsinki | 5th |
Question: In which venue did the athlete win the first place?
Let's think step by step.
The row with Position "1st" is row 1, whose venue is Paris.
Answer: Paris
""",
    "tfv": """Example 1:
| row_id | Player | Team | Goals |
|---|---|---|---|
| 0 | Smith | Reds | 12 |
| 1 | Jones | Blues | 15 |
| 2 | Brown | Reds | 9 |
Statement: The Reds scored more than 20 goals in total.
Let's think step by step.
Reds players: Smith 12, Brown 9. Total = 12 + 9 = 21 > 20.
Answer: True

Example 2:
| row_id | Date | Opponent | Result |
|---|---|---|---|
| 0 | May 1 | Lions | W 3-1 |
| 1 | May 8 | Tigers | L 0-2 |
Statement: The team won against the Tigers.
Let's think step by step.
Row 1 shows the result against the Tigers is "L 0-2", which is a loss.
Answer: False
""",
}

INIT = """{task_desc}

{fewshot}
Now it's your turn.
{table}
{qname}: {question}
Let's think step by step, then give the final answer in the last line as "Answer: <answer>".
"""

# --------------------------------------------------------------------------
# criticism of the initial CoT answer  (P_critic^{y0})
# --------------------------------------------------------------------------
CRITIC_INIT = """You are a strict reviewer of table-based reasoning. {task_desc}

{table}
{qname}: {question}

Candidate reasoning:
{reasoning}
Candidate answer: {answer}

Review the candidate following these rules:
1. Question Assessment: does the {qname_l} involve complex sub-questions, many rows, or long/noisy table content? If so, the relevant evidence may have been missed and refinement is needed.
2. Credibility and Correctness: check every cell value the reasoning cites against the table and every logical step. Any mistake means refinement is needed.
3. Handling of Computations: if the {qname_l} involves numerical reasoning (counting, sum, average, difference, comparison of numbers, ranking, dates arithmetic), refinement with tools is REQUIRED, because LLMs are unreliable at arithmetic.
3b. Boundary semantics: in counting/threshold questions, "more than N" is strictly > N, "at least N" is >= N; verify the comparison operator was applied exactly.
3c. Multi-part answers: verify EVERY part against the table; one wrong or extra part fails the answer.
3d. Minimal span: the final answer must be the minimal span (just the year/number/name) — units, venues, ranks, or qualifiers that the question did not ask for are errors.
3e. Aggregate rows: summary/total/remainder rows are not entities — never count or cite them as one.
4. If the answer is missing, not in the required format, or not grounded in the table, refinement is needed.
Only when the question is simple look-up, all cited values are correct and no computation is involved may you answer "No".

Output format (strictly):
Question Assessment: ...
Credibility and Correctness: ...
Handling of Computations: ...
Need refinement: Yes/No
Suggestions: <concrete suggestions for fixing the answer, or None>
"""

# --------------------------------------------------------------------------
# Tool 1: sub-table extraction
# --------------------------------------------------------------------------
SUBTABLE_SELECT = """You are given a table and a {qname_l}. Select the columns and rows that are relevant to answering it, to remove noisy content.
{task_desc}

{table}
{qname}: {question}
{suggestions}
Output a JSON object only, e.g. {{"columns": ["col A", "col B"], "rows": [0, 3, 5]}}.
Use the exact column names from the header. Use "rows": "all" if all rows may be relevant (e.g. counting or superlative questions). Keep every row that could matter.
"""

SUBTABLE_ANSWER = """{task_desc}

{fewshot}
Now it's your turn. The following is a sub-table extracted from the original table (row_id refers to the original row index) containing the relevant information.
{table}
{qname}: {question}
{suggestions}
Let's think step by step, then give the final answer in the last line as "Answer: <answer>".
"""

CRITIC_SUBTABLE = """You are a strict reviewer. A sub-table was extracted from the original table and used to answer the {qname_l}. {task_desc}

Original table:
{table}
{qname}: {question}

Extracted sub-table:
{subtable}
Reasoning on the sub-table:
{reasoning}
Answer: {answer}

Check:
1. Completeness: does the sub-table keep all columns and rows needed? Was any relevant row/column dropped?
2. Rationality: is the reasoning faithful to the cell values? Are there logic or arithmetic mistakes?
3. Correctness: does the answer actually answer the {qname_l} in the required format?

Output format (strictly):
Completeness: ...
Rationality: ...
Correctness: ...
Need refinement: Yes/No
Suggestions: <concrete suggestions, or None>
"""

# --------------------------------------------------------------------------
# Tool 2: python code generation
# --------------------------------------------------------------------------
PYTHON_GEN = """You are an expert Python data analyst. {task_desc}
The table is already loaded as a pandas DataFrame named `df` (all cells are strings; convert types yourself, handle commas, units, '$', '%', empty cells robustly).
{context}
Columns: {columns}
First rows of df (df has {n_rows} rows in total):
{table}
{qname}: {question}
{suggestions}
Write Python code that computes the answer and stores it in a variable named `answer`.{answer_type}
Do not print the whole table. Return only the code in a single ```python code block.
"""

CRITIC_PYTHON = """You are a strict code reviewer for table-based reasoning. {task_desc}

{table}
{qname}: {question}

Python code (the table is loaded as pandas DataFrame `df`, all cells are strings):
```python
{code}
```
Execution result: {exec_result}
Final answer extracted: {answer}

Check:
1. Rationality: does the code run without errors and set `answer`? Does it parse the cells correctly (types, units, empty values, string matching)? Does it filter the right rows/columns?
2. Correctness: does the executed answer correctly answer the {qname_l} according to the table, in the required format?

Output format (strictly):
Rationality: ...
Correctness: ...
Need refinement: Yes/No
Suggestions: <concrete suggestions to fix the code, or None>
"""

SUGGESTION_BLOCK = """
Your previous attempt:
{previous}
Reviewer suggestions to avoid the same mistakes:
{suggestions}
"""
