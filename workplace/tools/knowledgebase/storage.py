import os
from tools.knowledgebase.models import Chunk, List, asdict
import json

class Storage:
    def __init__(self, dir_path: str = "./chunk_storage"):
        self.storage_path = dir_path
        self.chunks_filename = os.path.join(dir_path, "chunks.jsonl")
        self.chunks: List[Chunk] = []
        os.makedirs(dir_path, exist_ok=True)
        self._load()
    
    def _load(self) -> None:
        if os.path.exists(self.chunks_filename):
            with open(self.chunks_filename, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.chunks = [Chunk(**x) for x in raw]
        else:
            self.chunks = []

    def add_chunk(self, chunk: Chunk):
        self.chunks.append(chunk)

    def save_chunks_to_json(self, chunks: List[Chunk]):
        chunks_dict = [asdict(chunk) for chunk in self.chunks]
        file = open(self.chunks_filename, 'w', encoding='utf-8')
        for chunk in chunks_dict:
            json_str = json.dumps(chunk, ensure_ascii=False)
            file.write(json_str + '\n')
        file.close()