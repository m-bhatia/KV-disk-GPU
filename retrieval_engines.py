from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import logging
import hnswlib
import pickle
import os

class EmbeddingService:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", chunk_size=None):
        self.model_name = model_name
        self.load_model(model_name)
        self.chunk_size = chunk_size

    def load_model(self, model_name=None):
        model_name = self.model_name if model_name is None else model_name
        self.model_name = model_name

        if model_name.startswith("sentence-transformers/"):
            self.model, self.tokenizer = SentenceTransformer(model_name), None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

    def create_embeddings(self, sentences):
        if self.tokenizer is None:
            return self.model.encode(sentences)
        else:
            inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).detach().numpy()
        
class RetrievalEngine:
    def __init__(self, **kwargs):
        self.data = kwargs.get("data")
        self.embedding_service = kwargs.get("embedding_service")
        self.separator = kwargs.get("separator", "\n")
        self.index_path = kwargs.get("index_path", None)
        self.k = kwargs.get("k", 3)

    def save_index(self, index_path):
        pass

    def load_index(self, index_path):
        pass

    def query(self, query: str) -> list:
        pass


class HNSWRetrievalEngine(RetrievalEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ef = kwargs.get("ef", 100)

        if self.ef <= self.k:
            raise ValueError("ef should always be greater than k")

        if self.index_path and os.path.exists(self.index_path):
            self.load_index(self.index_path)
        else:
            self.sentences = self.data.split(self.separator)
            self.sentence_embeddings = self.embedding_service.create_embeddings(self.sentences)
            self.dim = self.sentence_embeddings.shape[1]
            self.num_elements = self.sentence_embeddings.shape[0]
            self.index = hnswlib.Index(space="cosine", dim=self.dim)
            # self.index.init_index(max_elements=self.num_elements, ef_construction=100, M=16)
            self.index.init_index(ef_construction=100, M=16)
            self.index.add_items(self.sentence_embeddings)
            self.index.set_ef(self.ef)
            if self.index_path:
                self.save_index(self.index_path)

    def save_index(self, index_path):
        super().save_index(index_path)
        self.index.save_index(index_path)
        with open(index_path + "_sentences.pkl", "wb") as f:
            pickle.dump(self.sentences, f)

    def load_index(self, index_path):
        super().load_index(index_path)
        self.index = hnswlib.Index(
            space="cosine",
            dim=self.embedding_service.create_embeddings(["test"]).shape[1],
        )
        self.index.load_index(index_path)
        with open(index_path + "_sentences.pkl", "rb") as f:
            self.sentences = pickle.load(f)
        logging.info(f"Loaded index and sentences from {index_path}")

    def query(self, query: str) -> list:
        query_embedding = self.embedding_service.create_embeddings([query])
        labels, distances = self.index.knn_query(query_embedding, k=self.k)
        logging.info(f"Query labels: {labels}, distances: {distances}")
        return [self.sentences[label] for label in labels[0]]







