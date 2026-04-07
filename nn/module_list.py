"""ModuleList — PyTorch-compatible container of submodules."""

from __future__ import annotations

from typing import Any, Iterator

from .module import Module


class ModuleList(Module):
    """Holds submodules in a list. Acts like ``nn.ModuleList`` in PyTorch."""

    def __init__(self, modules: list[Module] | None = None):
        super().__init__()
        self._list: list[Module] = []
        if modules:
            for i, m in enumerate(modules):
                self._modules[str(i)] = m
                self._list.append(m)

    def __getitem__(self, idx: int | slice) -> Module | list[Module]:
        if isinstance(idx, slice):
            return self._list[idx]
        return self._list[idx]

    def __setitem__(self, idx: int, module: Module) -> None:
        self._modules[str(idx)] = module
        self._list[idx] = module

    def __len__(self) -> int:
        return len(self._list)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._list)

    def append(self, module: Module) -> "ModuleList":
        idx = len(self._list)
        self._modules[str(idx)] = module
        self._list.append(module)
        return self

    def extend(self, modules: list[Module]) -> "ModuleList":
        for m in modules:
            self.append(m)
        return self

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("ModuleList has no forward; iterate over submodules instead.")
