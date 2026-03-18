from code_filter.code_filter import Filter
from tools.knowledgebase.utils import transfer_dict_to_chunk
from tools.knowledgebase.storage import Storage

def extract_metadata_from_dataset_corpus(corpuses_number: int) -> list:
    code_filter = Filter("python")
    batch_code_info = list()

    for i in range(1, corpuses_number):
        code_filter.create_tree_from_file(f'corpuses_dataset/{i}.txt')
        batch_code_info.append(code_filter.extract_context(f'corpuses_dataset/{i}.txt'))

    return batch_code_info

def create_knowledgebase_from_dataset_corpus(corpuses_number: int):
    filtered_code_info = extract_metadata_from_dataset_corpus(corpuses_number=corpuses_number)

    storage = Storage()

    i = 1
    for x in filtered_code_info:
        chunk = transfer_dict_to_chunk(x, f'chunk_{i}', "fs")
        storage.add_chunk(chunk=chunk)
        i += 1

    storage.save_chunks_to_json(storage.chunks)


create_knowledgebase_from_dataset_corpus(100)
