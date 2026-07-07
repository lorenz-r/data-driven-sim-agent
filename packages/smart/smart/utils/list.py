from typing import List, Optional, TypeVar

_T = TypeVar("_T")


def safe_list_index(ls: List[_T], elem: _T) -> Optional[int]:
    try:
        return ls.index(elem)
    except ValueError:
        return None
