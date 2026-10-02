"""Disposable trusted-code worker. Protocol uses cloudpickle; not a security sandbox."""

import sys

import cloudpickle

from xuanji.execution import _python_repl_local, _serialize_state


def main():
    # Bound output files even if code writes directly to stdout/stderr.
    if sys.platform != "win32":
        import resource

        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    request = cloudpickle.load(sys.stdin.buffer)
    output, state, success = _python_repl_local(*request)
    payload = _serialize_state((output, state if success else None, success))
    if len(payload) > 64 * 1024 * 1024:
        payload = cloudpickle.dumps(
            ("RuntimeError: Python execution state exceeds 64 MiB", None, 0)
        )
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
