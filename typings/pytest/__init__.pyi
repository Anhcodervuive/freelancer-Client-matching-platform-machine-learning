# Minimal pytest stub so Pyright/Pylance can resolve the import in editors.
# Install the real dependency via `pip install -r requirements-dev.txt` for runtime.
from typing import Any, Callable, Iterable, Protocol

class _FixtureFunction(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

class FixtureFunctionMarker:
    def __call__(self, arg: Callable[..., Any] | None = None, *, name: str | None = None, scope: str | None = None, autouse: bool | None = None) -> Any: ...

fixture: FixtureFunctionMarker

class MarkDecorator:
    def __call__(self, obj: Callable[..., Any]) -> Callable[..., Any]: ...

class _Marks:
    def __getattr__(self, __name: str) -> MarkDecorator: ...

mark: _Marks

# Common test helpers
raises: Callable[..., Any]
skip: Callable[[str], None]
skipif: Callable[..., Any]
parametrize: Callable[..., Any]

def approx(__value: Any, *, rel: float | None = None, abs: float | None = None, nan_ok: bool = False) -> Any: ...

# Fixtures (minimal signatures for Pyright)
class MonkeyPatch:
    def setattr(self, obj: Any, name: str, value: Any, raising: bool = True) -> None: ...
    def setitem(self, mapping: Any, name: Any, value: Any) -> None: ...
    def delattr(self, obj: Any, name: str, raising: bool = True) -> None: ...
    def delitem(self, mapping: Any, name: Any) -> None: ...

monkeypatch: MonkeyPatch
