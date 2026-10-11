"""Tool registration: importing this package populates the default registry."""

from jiujiubiao.tools import tables as _tables  # noqa: F401 - registers table tools

__all__: list[str] = []
