"""Nhẹ nhàng mô phỏng pytest.approx để chạy file test trực tiếp."""
from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Sequence


@dataclass
class _Approx:
    target: object
    rel: float | None
    abs: float | None
    nan_ok: bool

    def _is_nan(self, value: object) -> bool:
        return isinstance(value, float) and value != value

    def _compare(self, actual: object, expected: object) -> bool:
        if self.nan_ok and (self._is_nan(actual) or self._is_nan(expected)):
            return self._is_nan(actual) and self._is_nan(expected)

        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return isclose(actual, expected, rel_tol=self.rel or 0.0, abs_tol=self.abs or 0.0)

        if isinstance(actual, Sequence) and isinstance(expected, Sequence):
            if len(actual) != len(expected):
                return False
            return all(self._compare(a, e) for a, e in zip(actual, expected))

        return actual == expected

    def __eq__(self, other: object) -> bool:  # pragma: no cover - logic đã được dùng gián tiếp
        return self._compare(other, self.target)

    def __repr__(self) -> str:  # pragma: no cover - chỉ hỗ trợ debug
        return f"approx({self.target!r}, rel={self.rel}, abs={self.abs}, nan_ok={self.nan_ok})"


def approx(__value: object, *, rel: float | None = None, abs: float | None = None, nan_ok: bool = False) -> _Approx:
    return _Approx(__value, rel=rel, abs=abs, nan_ok=nan_ok)


__all__ = ["approx"]
