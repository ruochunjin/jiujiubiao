# coding=utf-8
# Solver prompt templates, adapted from the TALON repository (Apache-2.0,
# Copyright 2024 The Google Research Authors) with task routing removed.
"""ReAct solver prompts for the three supported task styles."""

pyreact_solve_table_prompt_tools = """
Given a large table and several sampled rows, indicated that the table have more than 50 rows, you need to answer the question: "{query}".
The table is a pandas dataframe in Python. The name of the dataframe is `df`. Your task is to use tools to answer the question.
- `get_column_meaning()`: This tool can help you understand the most relevant columns and column meanings about the question.
- `find_columns_containing_value_fuzzy(value)`: If you are searching for a specific value that differs from the column name, use this tool to identify potential variations of the value across columns. This is useful when the value may have different representations or may not match exactly with what is provided in the problem.
- `find_column_format(column)`: Find or Process the format of a specific column. This is useful when you need to find or process the value of a column.
- `get_value(value)`: Search for a specific value. This will help us determine if the value exists within a column.
- `get_row(row_index)`: Use this tool to retrieve a specific row from the table. Or you can use this tool to understand the relationship of two colums in a row. Be carful to use it until you exact know the index of the row, do not assume the index of the row.\x20
- `def()`: This tool can generate Python code,you should generate with the format of def()```python\n[### the code you generate ###]\n``` to fetch data or calculate or other actions. If you define a variable, you must print it to show the output.Do not output any additional text. Ensure that the code is self-contained and executable. Do not sample values.

Strictly follow the given format to respond:
Thought: you should always print about the thinking process about what to do based on the previous observation.
Action: Directly output the tool you choosed to use, do not output other inforamtion.
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: before giving the final answer, you should think about the observations

Final Answer: the final answer to the original input question, only print the answer,the answer should not be a variable, no other form.

Notes:

- Do not use markdown or any other formatting in your responses.
- Ensure the last line is only "Final Answer: answer".
- Directly output the Final Answer rather than outputting by Python.
- Ensure to have a concluding thought that verifies the table, observations and the statement before giving the final answer.
- You can call a same tool multiple times simultaneously. For example, you can output Action: tool1(args1), tool1(args2) at one same time.
- If you use def() tool, Do not sample data, just process the dataframe.\x20

Now, given a table , please answer the question: "{query}".
{table}
The table is a pandas dataframe in Python. The name of the dataframe is `df`. Your task is think step by step, use tools to answer the question based on the table.

Begin!
"""


pyreact_solve_table_prompt_tools_tfv = """
Given a large table regarding "{table_caption}", you need to verify the statement: "{query}".
The table is a pandas dataframe in Python. The name of the dataframe is `df`. Your task is to use tools to verify the statement.
- `get_column_meaning()`: This tool can help you understand the most relevant columns and column meanings about the question.
- `find_columns_containing_value_fuzzy(value)`: If you are searching for a specific value that differs from the column name, use this tool to identify potential variations of the value across columns. This is useful when the value may have different representations or may not match exactly with what is provided in the problem.
- `find_column_format(column)`: Find or Process the format of a specific column. This is useful when you need to find or process the value of a column.
- `get_value(value)`: Search for a specific value. This will help us determine if the value exists within a column.
- `get_row(row_index)`: Use this tool to retrieve a specific row from the table. Or you can use this tool to understand the relationshio of two colums in a row. Be carful to use it until you exact know the index of the row, do not assume the index of the row.\x20
- `def()`: This tool can generate Python code,you should generate with the format of def()```python\n[### the code you generate ###]\n``` to fetch data or calculate or other actions. If you define a variable, you must print it to show the output.Do not output any additional text. Ensure that the code is self-contained and executable. Do not sample values.

Strictly follow the given format to respond:
Thought: you should always print about the thinking process about what to do based on the previous observation.
Action: Directly output the tool you choosed to use, do not output other inforamtion.
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: before giving the final answer, you should think about the observations

Final Answer: the final answer to the original input question, only print the answer,the answer should not be a variable, no other form.

Notes:

- Do not use markdown or any other formatting in your responses.
- Ensure the last line is only "Final Answer: true/false".
- Directly output the Final Answer rather than outputting by Python.
- Ensure to have a concluding thought that verifies the table, observations and the statement before giving the final answer.
- You can call a same tool multiple times simultaneously. For example, you can output Action: tool1(args1), tool1(args2) at one same time.
- If you use def() tool, Do not sample data, just process the dataframe.\x20
- You have to make sure gather enough information to answer the question.\x20

Now, given a table regarding "{table_caption}", please verify the statement: "{query}".

{table}

Begin!
"""


pyreact_solve_table_prompt_tools_db = """
Given a large table regarding "{table_caption}", you need to answer the question: "{query}".
The table is a pandas dataframe in Python. The name of the dataframe is `df`. Your task is to use tools to answer the question.
- `get_column_meaning()`: This tool can help you understand the most relevant columns and column meanings about the question.
- `find_columns_containing_value_fuzzy(value)`: If you are searching for a specific value that differs from the column name, use this tool to identify potential variations of the value across columns. This is useful when the value may have different representations or may not match exactly with what is provided in the problem.
- `find_column_format(column)`: Find the format of a specific column. This is useful when you need to understand the data type or structure of a particular column in the table.
- `get_value(value)`: Search for a specific value. This will help us determine if the value exists within a column.
- `get_row(row_index)`: Use this tool to retrieve a specific row from the table. Or you can use this tool to understand the relationshio of two colums in a row. Be carful to use it until you exact know the index of the row, do not assume the index of the row.\x20

Strictly follow the given format to respond:
Thought: you should always print about the thinking process about what to do based on the previous observation.
Action: Directly output the tool you choosed to use, do not output other inforamtion.
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: before giving the final answer, you should think about the observations

Final Answer: the final answer to the original input question, only print the answer,the answer should not be a variable, no other form.

Notes:

- Do not use markdown or any other formatting in your responses.
- Ensure the last line is only "Final Answer: answer".
- Directly output the Final Answer rather than outputting by Python.
- Ensure to have a concluding thought that verifies the table, observations and the statement before giving the final answer.
- You can call a same tool multiple times simultaneously. For example, you can output Action: tool1(args1), tool1(args2) at one same time.
- Do not generate any code, just use the tools.

Now, given a table regarding "{table_caption}", please answer the question: "{query}", here is a evidence to help you answer the question: "{evidence}".
{table}
The table is a pandas dataframe in Python. The name of the dataframe is `df`. Your task is think step by step, use tools to answer the question based on the table.

Begin!
"""


"""Prompts for table description and column format processing."""

DESCRIPTION_PROMPT = """
Given a partial table preview along with a thought, focusing only on information relevant to the question and the thought, analyze the table and extract only the most relevant information.
Please Note:
- the given rows only contain a small portion of the table, not indicate the entire table.
- Directly output the most relevant columns that might be related to the question and the thought, and the column meaning of a column.
Output Format:
Directly output the columns and column meaning without any additional text or explanations.
Column name: Column Meaning
eg:
Column1: Column Meaning1
Column2: Column Meaning2
Table preview:\n{preview_data}\nQuestion:{question} Thought:{thought}.
        """

FORMAT_PROMPT = """
The column type of '{column_name}' is {column_dtype}.
Here are some example values from the column: {sampled_values}.
You should determine whether there is a need to process the column using Python code for the given purpose.\x20
Determine whether the given column needs processing using Python based on its content and format.
The column can be processed by df["{column_name}"]
Purpose: {thought}
Notes:
If the column's format needs additional transformation to calculate the result based on the purpose, then it needs to be processed and transformed into the correct type.
For example:
- If sampled_values are ['1,200,300', '1,300,299'], they need to be processed to ['1200300', '1300299'].
- Use df to process the column, and output the code in the format of:
```python
# your code here
```
- Do not output any additional text or create any sample data.
- Do not sample DataFrame or generate any additional data. Only generate code to process the column.
- The sampled values are not the entire table, just a few examples of the column's data.
- Just process the format of the column, do not do additional operations, such as calculating the mean or sum.
- If the column's format is already correct, just return "[[No]]".
"""

TASK_TEMPLATES = {
    "wtq": pyreact_solve_table_prompt_tools,
    "tabfact": pyreact_solve_table_prompt_tools_tfv,
    "bird": pyreact_solve_table_prompt_tools_db,
}
