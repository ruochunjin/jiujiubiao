"""Table representation helpers."""
import re
from pathlib import Path

try:
    import pandas as pd
except ImportError:  # pandas only needed for the python tool
    pd = None


class Table:
    """A simple table: header + rows (all cells as strings), optional title/context."""

    def __init__(self, header, rows, title="", context=""):
        self.header = [str(h).strip() for h in header]
        self.rows = [[("" if c is None else str(c)).strip() for c in r] for r in rows]
        self.title = title or ""
        self.context = context or ""   # extra text (FinQA / TabMWP style)

    @classmethod
    def from_obj(cls, obj, title="", context=""):
        """Accepts {'header','rows'} | [[header],[row],...] | pandas DataFrame | markdown/csv-ish str."""
        if isinstance(obj, Table):
            return obj
        if pd is not None and isinstance(obj, pd.DataFrame):
            return cls(list(obj.columns), obj.astype(str).values.tolist(), title, context)
        if isinstance(obj, dict):
            return cls(obj["header"], obj["rows"], obj.get("title", title), obj.get("context", context))
        if isinstance(obj, list):
            return cls(obj[0], obj[1:], title, context)
        if isinstance(obj, str):
            # A short newline-free string naming an existing file is a PATH
            # (csv/tsv); anything else is markdown/csv-ish content.
            if "\n" not in obj and len(obj) < 260 and Path(obj).exists():
                return cls.from_csv(obj, title, context)
            return cls.from_text(obj, title, context)
        raise TypeError(f"unsupported table type {type(obj)}")

    @classmethod
    def from_csv(cls, path, title="", context=""):
        """Load a CSV/TSV file (pandas when available, csv module otherwise)."""

        import csv as csv_module

        sep = "	" if str(path).endswith((".tsv", ".tab")) else ","
        with open(path, encoding="utf-8", errors="replace", newline="") as handle:
            rows = [[c.strip() for c in row] for row in csv_module.reader(handle, delimiter=sep) if row]
        if not rows:
            raise ValueError(f"empty table file: {path}")
        return cls(rows[0], rows[1:], title or Path(path).stem, context)

    @classmethod
    def from_text(cls, text, title="", context=""):
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        lines = [line for line in lines if not re.fullmatch(r"\|?[\s:\-|]+\|?", line)]
        sep = "|" if "|" in lines[0] else ("\t" if "\t" in lines[0] else ",")
        rows = [[c.strip() for c in line.strip("|").split(sep)] for line in lines]
        return cls(rows[0], rows[1:], title, context)

    # ---- serialisation -------------------------------------------------
    def to_markdown(self, max_rows=None):
        rows = self.rows if max_rows is None else self.rows[:max_rows]
        out = []
        if self.title:
            out.append(f"Table title: {self.title}")
        out.append("| row_id | " + " | ".join(self.header) + " |")
        out.append("|" + "---|" * (len(self.header) + 1))
        for i, r in enumerate(rows):
            r = (r + [""] * len(self.header))[:len(self.header)]
            out.append(f"| {i} | " + " | ".join(c.replace("|", "/").replace("\n", " ") for c in r) + " |")
        if max_rows is not None and len(self.rows) > max_rows:
            out.append(f"... ({len(self.rows) - max_rows} more rows)")
        s = "\n".join(out)
        if self.context:
            s = f"Context: {self.context}\n\n" + s
        return s

    def to_df(self):
        if pd is None:
            raise ImportError("pandas is required")
        cols, seen = [], {}
        for h in self.header:  # de-duplicate column names
            k = h or "col"
            if k in seen:
                seen[k] += 1
                k = f"{k}_{seen[k]}"
            else:
                seen[k] = 0
            cols.append(k)
        rows = [(r + [""] * len(cols))[:len(cols)] for r in self.rows]
        return pd.DataFrame(rows, columns=cols)

    def to_payload(self):
        return {"header": self.to_df().columns.tolist(), "rows": [
            (r + [""] * len(self.header))[:len(self.header)] for r in self.rows]}

    def sub_table(self, columns=None, row_ids=None):
        idx = list(range(len(self.header)))
        if columns:
            low = [h.lower() for h in self.header]
            keep = []
            for c in columns:
                c = str(c).strip().lower()
                if c in low:
                    keep.append(low.index(c))
            if keep:
                idx = sorted(set(keep))
        rows = self.rows
        if row_ids:
            ids = [int(i) for i in row_ids if str(i).strip().lstrip("-").isdigit()]
            ids = [i for i in ids if 0 <= i < len(self.rows)]
            if ids:
                rows = [self.rows[i] for i in sorted(set(ids))]
        new_rows = [[(r + [""] * len(self.header))[i] for i in idx] for r in rows]
        return Table([self.header[i] for i in idx], new_rows, self.title, self.context)

    def __len__(self):
        return len(self.rows)
