import os
from tools.knowledgebase.models import Chunk, List, asdict
import json


# прини

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
                raw = [json.loads(line) for line in f if line.strip()]
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

    def get_filtered_chunks(self, language: str, imports: list[str]) -> set:

        filtered = self.chunks

        indexes = list()

        # обязательный фильтр - язык
        if language is not None:
            filtered = [c for c in filtered if c.language == language]

        # обязательный фильтр - импорты
        if imports:
            imports_set = set(imports)
            filtered = [
                c for c in filtered
                if imports_set.intersection(c.imports)
            ]

        # classes и functions — игнорируются (мягкие фильтры)

        for x in filtered:
            indexes.append(int(x.chunk_id))

        return set(indexes)