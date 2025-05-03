from retrieval_engines import EmbeddingService, HNSWRetrievalEngine
from datasets import load_dataset
from tqdm import tqdm
import os
import sys

def create_index(embedding_service, index_path):
    ds = load_dataset("community-datasets/wiki_snippets", "wiki40b_en_100_0")
    hnsw_index = HNSWRetrievalEngine(
        data="",
        embedding_service=embedding_service,
        separator="\n",
        k=5,
        ef=100,
        index_path=None,
        max_elements=len(ds["train"]),
    )
    for i in tqdm(range(100000)):
        hnsw_index.add_item(ds["train"][i]["passage_text"])
    hnsw_index.save_index(index_path)


def augment_query(query):
    embedding_service = EmbeddingService(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_path = os.path.join(os.getcwd(), "wiki_index_small.hnsw")
    
    if not os.path.isfile(index_path):
        print("Creating index...")
        create_index(embedding_service, index_path)
    
    print("Loading index...")
    hnsw_index = HNSWRetrievalEngine(
        data="",
        embedding_service=embedding_service,
        index_path=index_path,
    )

    return hnsw_index.query(query)


if __name__ == "__main__":
    assert(len(sys.argv) == 2)

    results = augment_query(sys.argv[1])
    for result in results:
        print()
        print(result)
