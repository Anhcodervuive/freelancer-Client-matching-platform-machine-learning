"""Runtime shim for pytest so editors resolve the module while using stubs."""

from typing import Any


def approx(__value: Any, *, rel: float | None = None, abs: float | None = None, nan_ok: bool = False) -> Any:
    """Placeholder for pytest.approx; real behavior comes from pytest when installed."""
    return __value


__all__ = [
    "approx",
]
