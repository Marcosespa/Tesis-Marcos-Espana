from typing import List
from dataclasses import dataclass

@dataclass
class CleanConfig:
    drop_empty: bool = True
    normalize_whitespace: bool = True
    fix_hyphens: bool = True

