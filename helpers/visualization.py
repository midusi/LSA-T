from typing import Callable, TypeVar, Iterable
from pathlib import Path

from pandas import Index


def save_fig(fig, name: str, path: Path):
    fig.get_figure().savefig(path.resolve() / name, bbox_inches = 'tight')

def get_sort_key(other_symbol: str, other_val: int):
    'Return function to sort elements of index replacing other_symbol with other_val'
    def sort_key(i: Index):
        return list(map(lambda i: int(i) if i != other_symbol else other_val, i))
    return sort_key

T_Items = TypeVar("T_Items")

def group_items(
        iter: Iterable[T_Items],
        group_key: Callable[[T_Items], bool],
        other_symbol: T_Items
    ) -> list[T_Items]:
    'Groups elements of a counter if group_key is true for its value'
    return [(v if not group_key(v) else other_symbol) for v in iter]
