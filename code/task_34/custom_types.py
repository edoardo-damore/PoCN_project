from enum import Enum
from typing import Any

type ResultsDict = dict[int, dict[str, Any]]


class UGStrategy(Enum):
    EMP = 0
    PRG = 1
    RND = 2


class WPDStrategy(Enum):
    C = 0
    D = 1


class UpdateRule(Enum):
    REP = 0
    UI = 1
    MOR = 2
