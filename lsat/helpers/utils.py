from typing import TypeVar, Callable, Iterable


A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")

def compose(f: Callable[[B], C], g: Callable[[A], B]) -> Callable[[A], C]:
    return lambda *a: f(g(*a))

T = TypeVar("T")
def flatten(list: Iterable[Iterable[T]]) -> list[T]:
    return [item for sublist in list for item in sublist]
