# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from copy import deepcopy
from io import BytesIO, StringIO
from pathlib import Path
from types import FunctionType, ModuleType

import cloudpickle
import pandas as pd


def parse_code_from_string(input_string):
    """
    Parse executable code from a string, handling various markdown-like code block formats.

    Parameters:
    input_string (str): The input string.

    Returns:
    str: The parsed code.
    """

    # Pattern to match code blocks wrapped in triple backticks, with optional language specification
    triple_backtick_pattern = r"```(?:python[ \t]*\n|[ \t]*\n)?(.*?)```"
    match = re.search(triple_backtick_pattern, input_string, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Pattern to match code blocks wrapped in single backticks
    single_backtick_pattern = r"`(.*?)`"
    match = re.search(single_backtick_pattern, input_string, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    # Default return if no code block patterns are matched
    return input_string.strip()


def copy_execution_state(state):
    """Copy table/variable state and bind saved Python functions to the copied globals.

    Modules remain shared runtime dependencies. This does not isolate external
    side effects such as file writes, network calls or changes to module state.
    """
    result = {}
    memo = {id(state): result}
    pending_functions = []
    pending_frames = []
    closure_cells = {}
    seen = set()

    def prepare(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, ModuleType):
            memo[id(value)] = value
        elif isinstance(value, FunctionType) and (
            value.__globals__ is state or value.__module__ == "__xuanji_exec__"
        ):
            cells = None
            if value.__closure__:
                cells = tuple(
                    closure_cells.setdefault(id(cell), (lambda x: lambda: x)(None).__closure__[0])
                    for cell in value.__closure__
                )
            copied = FunctionType(value.__code__, result, value.__name__, None, cells)
            copied.__module__ = value.__module__
            memo[id(value)] = copied
            pending_functions.append((value, copied))
            prepare(value.__defaults__)
            prepare(value.__kwdefaults__)
            prepare(value.__dict__)
            if value.__closure__:
                for cell in value.__closure__:
                    try:
                        prepare(cell.cell_contents)
                    except ValueError:
                        pass
        elif isinstance(value, (pd.DataFrame, pd.Series)):
            copied = value.copy(deep=True)
            memo[id(value)] = copied
            pending_frames.append((value, copied))
            # Pandas deep copies do not recursively copy Python objects in cells.
            columns = (
                [value]
                if isinstance(value, pd.Series)
                else [value.iloc[:, i] for i in range(value.shape[1])]
            )
            for column in columns:
                if column.dtype == object:
                    for item in column:
                        prepare(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                prepare(key)
                prepare(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                prepare(item)

    if "__builtins__" in state:
        memo[id(state["__builtins__"])] = state["__builtins__"]
        seen.add(id(state["__builtins__"]))
    for value in state.values():
        prepare(value)
    for original, copied in pending_frames:
        if isinstance(original, pd.Series):
            if original.dtype == object:
                for i in range(len(original)):
                    copied.iat[i] = deepcopy(original.iat[i], memo)
        else:
            for j, dtype in enumerate(original.dtypes):
                if pd.api.types.is_object_dtype(dtype):
                    for i in range(len(original)):
                        copied.iat[i, j] = deepcopy(original.iat[i, j], memo)
    for key, value in state.items():
        result[key] = deepcopy(value, memo)
    for original, copied in pending_functions:
        copied.__defaults__ = deepcopy(original.__defaults__, memo)
        copied.__kwdefaults__ = deepcopy(original.__kwdefaults__, memo)
        copied.__dict__.update(deepcopy(original.__dict__, memo))
        if original.__closure__:
            for source, target in zip(original.__closure__, copied.__closure__):
                try:
                    target.cell_contents = deepcopy(source.cell_contents, memo)
                except ValueError:
                    del target.cell_contents
    return result


class _StatePickler(cloudpickle.CloudPickler):
    """Keep shared nonlocal cells across the worker round trip."""

    def __init__(self, file):
        super().__init__(file)
        self._shared_cells = {}

    def _function_getnewargs(self, func):
        # Cloudpickle normally allocates separate empty cells for each function.
        # Reuse those placeholders for functions that originally shared a cell.
        code, globals_, name, defaults, cells = super()._function_getnewargs(func)
        if cells is not None:
            cells = tuple(
                self._shared_cells.setdefault(id(original), empty)
                for original, empty in zip(func.__closure__, cells)
            )
        return code, globals_, name, defaults, cells


def _serialize_state(value):
    stream = BytesIO()
    _StatePickler(stream).dump(value)
    return stream.getvalue()


class _LimitedOutput(StringIO):
    def write(self, text):
        if self.tell() + len(text) > 100_000:
            raise RuntimeError("Python output exceeds 100000 characters")
        return super().write(text)


def python_repl_ast(code, custom_globals=None, custom_locals=None, memory=None):
    """Execute in a disposable child process with a wall-clock deadline.

    Success returns the committed state; errors/timeouts return the original
    state. This is execution isolation, NOT a file/network permission sandbox.
    Serialized state is trusted Python data, not a safe untrusted-input format.
    """
    memory = {} if memory is None else memory
    process = None
    try:
        timeout = float(os.environ.get("XUANJI_PYTHON_TIMEOUT", "10"))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("XUANJI_PYTHON_TIMEOUT must be a finite positive number")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("No Python code provided")
        payload = _serialize_state((code, custom_globals, custom_locals, memory))
        if len(payload) > 64 * 1024 * 1024:
            raise ValueError("Python execution state exceeds 64 MiB")
        # No API keys or inherited provider settings are needed by generated code.
        # Windows Python needs SystemRoot (and TMP/TEMP for tempfile) even for
        # a bare interpreter start — stripping them kills the worker with a
        # bare exit 1 on CPython ≤ 3.12 (CI finding, windows-latest).
        env = {
            "PATH": os.defpath,
            "PYTHONDONTWRITEBYTECODE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        if os.name == "nt":
            for var in ("SystemRoot", "TMP", "TEMP", "USERPROFILE", "APPDATA"):
                if os.environ.get(var):
                    env[var] = os.environ[var]
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            source_root = str(Path(__file__).resolve().parents[1])
            worker_start = (
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "from xuanjitu.execution_worker import main; main()"
            )
            process = subprocess.Popen(
                [sys.executable, "-I", "-c", worker_start, source_root],
                stdin=subprocess.PIPE,
                stdout=output,
                stderr=errors,
                env=env,
                cwd=tempfile.gettempdir(),
                start_new_session=(os.name == "posix"),
            )
            try:
                process.communicate(payload, timeout=timeout)
            except subprocess.TimeoutExpired:
                _stop_worker(process)
                return f"TimeoutError: Python execution exceeded {timeout:g} seconds", memory, 0
            if process.returncode != 0:
                return (
                    f"RuntimeError: Python worker exited with code {process.returncode}",
                    memory,
                    0,
                )
            output.seek(0)
            response = output.read(64 * 1024 * 1024 + 1)
            if len(response) > 64 * 1024 * 1024:
                raise ValueError("Python execution state exceeds 64 MiB")
            observation, state, success = cloudpickle.loads(response)
            if not isinstance(observation, str) or success not in (0, 1):
                raise ValueError("Invalid Python worker response")
            if success and not isinstance(state, dict):
                raise ValueError("Invalid Python execution state")
            return observation, state if success else memory, success
    except Exception as error:
        return f"{type(error).__name__}: {error}", memory, 0
    finally:
        if process is not None:
            # Also reap subprocesses that generated code left running.
            _stop_worker(process)


def _stop_worker(process):
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
    process.wait()


def _python_repl_local(code, custom_globals=None, custom_locals=None, memory=None):
    """Worker-only transactional execution; never call for model code in the parent."""
    memory = {} if memory is None else memory
    output = _LimitedOutput()
    try:
        # Saved state takes precedence over initial bindings (especially df).
        namespace = copy_execution_state(memory)
        initial = {**(custom_globals or {}), **(custom_locals or {})}
        missing = {key: value for key, value in initial.items() if key not in namespace}
        namespace.update(copy_execution_state(missing))
        namespace["__name__"] = "__xuanji_exec__"
        if not isinstance(code, str) or not code.strip():
            raise ValueError("No Python code provided")
        tree = ast.parse(code)
        with redirect_stdout(output):
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                prefix = ast.Module(body=tree.body[:-1], type_ignores=[])
                exec(compile(prefix, "<xuanji>", "exec"), namespace, namespace)
                expression = ast.Expression(body=tree.body[-1].value)
                value = eval(compile(expression, "<xuanji>", "eval"), namespace, namespace)
                if value is not None:
                    print(object_to_string(value, code.strip()), end="")
            else:
                exec(compile(tree, "<xuanji>", "exec"), namespace, namespace)
        return output.getvalue(), namespace, 1
    except Exception as error:
        return f"{type(error).__name__}: {error}", memory, 0


def object_to_string(obj, command):
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, pd.DataFrame):
        if len(obj) == 0:
            return "Empty DataFrame"
    elif command == "df.columns":
        obj = obj.tolist()
        if len(obj) > 20:
            return str(obj[:10])[:-1] + ", ..., " + str(obj[-10:])[1:]
    return str(obj)
