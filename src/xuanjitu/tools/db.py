"""Database tools (RAISR family) over a sqlite3 backend.

``Database`` wraps any SQLite database file, or builds an in-memory one from
a pandas DataFrame — which is also how XuanJi bridges the table world and
the SQL world: register these tools when the question targets a database,
and the solver gets schema-aware SQL tools instead of DataFrame tools.
"""

from __future__ import annotations

import difflib
import sqlite3

import pandas as pd

from xuanjitu.registry import ToolContext, register_tool


class Database:
    """Minimal SQLite-backed database handle."""

    def __init__(self, connection: sqlite3.Connection, db_id: str = "db") -> None:
        self.conn = connection
        self.db_id = db_id

    @classmethod
    def from_sqlite(cls, path: str, db_id: str | None = None) -> "Database":
        return cls(sqlite3.connect(path), db_id or str(path))

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, table_name: str = "t") -> "Database":
        conn = sqlite3.connect(":memory:")
        df.to_sql(table_name, conn, index=False, if_exists="replace")
        return cls(conn, table_name)

    def tables(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [row[0] for row in rows]

    def columns(self, table: str) -> list[tuple[str, str]]:
        try:
            return [(row[1], row[2]) for row in self.conn.execute(f'PRAGMA table_info("{table}")')]
        except sqlite3.Error:
            return []

    def schema_text(self, sample_values: int = 0) -> str:
        lines = []
        for table in self.tables():
            cols = ", ".join(f"{name} {dtype}" for name, dtype in self.columns(table))
            lines.append(f"TABLE {table} ({cols})")
        return "\n".join(lines)

    def execute(self, sql: str, limit: int = 50) -> str:
        """Run a read-only query and render the first ``limit`` rows."""

        try:
            cursor = self.conn.execute(sql)
        except sqlite3.Error as error:
            return f"SQL Error: {error}"
        rows = cursor.fetchmany(limit)
        header = [d[0] for d in cursor.description or []]
        if not rows:
            return "Empty result."
        body = "\n".join(" | ".join(str(cell) for cell in row) for row in rows)
        suffix = ""
        if cursor.fetchone() is not None:
            suffix = f"\n... (truncated to {limit} rows)"
        return " | ".join(header) + "\n" + body + suffix

    def search_value(self, value: str, top_k: int = 5) -> list[str]:
        """Find columns whose cells contain ``value`` (exact or substring)."""

        hits = []
        for table in self.tables():
            for column, _dtype in self.columns(table):
                try:
                    rows = self.conn.execute(
                        f'SELECT DISTINCT "{column}" FROM "{table}" WHERE CAST("{column}" AS TEXT) LIKE ? LIMIT ?',
                        (f"%{value}%", top_k),
                    ).fetchall()
                except sqlite3.Error:
                    continue
                for (cell,) in rows:
                    hits.append(f"{table}.{column} contains '{cell}'")
        return hits[:top_k]


@register_tool(name="execute_sql", takes_arg=True)
def execute_sql(ctx: ToolContext, query: str) -> str:
    """execute_sql(sql) — Execute a SQL query against the database and return the result rows."""
    if ctx.db is None:
        return "Error: no database attached to this question."
    return ctx.db.execute(str(query))


@register_tool(name="find_schema_fuzzy", takes_arg=True)
def find_schema_fuzzy(ctx: ToolContext, value: str) -> str:
    """find_schema_fuzzy(value) — Fuzzy-match a literal against schema (tables/columns) and cell values; use it to ground question phrases."""
    if ctx.db is None:
        return "Error: no database attached to this question."
    value = str(value)
    names = [name for table in ctx.db.tables() for name in [table] + [c for c, _ in ctx.db.columns(table)]]
    close = difflib.get_close_matches(value, names, n=5, cutoff=0.5)
    value_hits = ctx.db.search_value(value)
    parts = []
    if close:
        parts.append("Schema matches: " + ", ".join(close))
    if value_hits:
        parts.append("Value matches:\n" + "\n".join(value_hits))
    return "\n".join(parts) if parts else f"No schema or value matches for '{value}'."


@register_tool(name="is_value_in_column", takes_arg=True)
def is_value_in_column(ctx: ToolContext, value: str) -> str:
    """is_value_in_column(table.column=value) — Check whether a column contains a value; returns matching rows."""
    if ctx.db is None:
        return "Error: no database attached to this question."
    raw = str(value)
    if "=" in raw:
        target, _, needle = raw.partition("=")
    else:
        target, _, needle = raw.rpartition(":")
    table, _, column = target.strip().partition(".")
    needle = needle.strip().strip("'\"")
    if not (table and column):
        return "Usage: is_value_in_column(table.column=value)"
    try:
        rows = ctx.db.conn.execute(
            f'SELECT * FROM "{table}" WHERE CAST("{column}" AS TEXT) = ? LIMIT 3', (needle,)
        ).fetchall()
    except sqlite3.Error as error:
        return f"SQL Error: {error}"
    if not rows:
        return f"'{needle}' NOT found in {table}.{column}."
    return f"'{needle}' found in {table}.{column}; {len(rows)} sample row(s): {rows}"


@register_tool(name="get_column_info", takes_arg=True)
def get_column_info(ctx: ToolContext, value: str) -> str:
    """get_column_info(table.column) — Return the column's type and sample distinct values."""
    if ctx.db is None:
        return "Error: no database attached to this question."
    table, _, column = str(value).partition(".")
    if not (table and column):
        return "Usage: get_column_info(table.column)"
    dtypes = dict(ctx.db.columns(table))
    if column not in dtypes:
        return f"Column '{column}' not found in table '{table}'. Available: {list(dtypes)}"
    try:
        values = ctx.db.conn.execute(
            f'SELECT DISTINCT "{column}" FROM "{table}" LIMIT 10'
        ).fetchall()
    except sqlite3.Error as error:
        return f"SQL Error: {error}"
    return f"{table}.{column} ({dtypes[column]}); sample values: {[v[0] for v in values]}"


__all__ = ["Database"]
