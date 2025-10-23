"""Improved version of MyZoo without modifying the original file.

Improvements:
- Avoid mutable default argument for animals
- Keep comparisons side-effect free (don't sort internal lists in-place)
- Provide __repr__, type annotations and docstrings
- Fix the __main__ guard and include a small demo with assertions
"""

from __future__ import annotations
from typing import Dict


class MyZoo:
    """A simple container for animals (species -> count).

    Equality is defined by species set (counts ignored). Length is total animals.
    """

    def __init__(self, animals: Dict[str, int] | None = None) -> None:
        animals = dict(animals) if animals is not None else {}
        self.animals: Dict[str, int] = animals

    def __repr__(self) -> str:  # useful for debugging
        items = ", ".join(f"{k}={v}" for k, v in self.animals.items())
        return f"MyZoo({{{items}}})"

    def __str__(self) -> str:
        parts = [f"{species}-{count}" for species, count in self.animals.items()]
        return " ".join(parts)

    def __len__(self) -> int:
        return sum(self.animals.values())

    def __eq__(self, another: object) -> bool:
        if not isinstance(another, MyZoo):
            return NotImplemented
        return set(self.animals.keys()) == set(another.animals.keys())

    def __ne__(self, another: object) -> bool:
        eq = self.__eq__(another)
        if eq is NotImplemented:
            return NotImplemented
        return not eq


if __name__ == "__main__":
    z1 = MyZoo({"dog": 5, "cat": 7})
    z2 = MyZoo({"cat": 3, "dog": 10})
    z3 = MyZoo({"rabbit": 2})

    print("Zoo1:", z1)
    print("Zoo2:", z2)
    print("Zoo3:", z3)
    print("Len Zoo1:", len(z1))
    print("Zoo1 == Zoo2:", z1 == z2)
    print("Zoo1 == Zoo3:", z1 == z3)

    # Basic self-checks
    assert len(z1) == 12
    assert z1 == z2
    assert z1 != z3
    print("Demo passed.")
