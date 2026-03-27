from tree_sitter_go import language

from tools.knowledgebase.models import Chunk
from code_filter.filter_models import CodeInfo

def transfer_dict_to_chunk(d: dict, chunk_id: str, content: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        language=d['language'],
        imports=d['imports'],
        classes=d['classes'],
        functions=d['functions'],
        content=content
    )