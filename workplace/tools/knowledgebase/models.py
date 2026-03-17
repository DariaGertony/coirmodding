from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple, TypedDict

from tree_sitter_go import language

@dataclass
class Chunk:
    chunk_id: str
    language: str
    imports: List[str]
    classes: List[str]
    functions: List[str]
    content: str
