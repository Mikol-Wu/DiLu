import os
import numpy as np
from typing import List, Tuple

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document

try:
    from langchain.embeddings import HuggingFaceEmbeddings  # langchain>=0.0.3xx
except Exception:
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from dilu.scenario.envScenario import EnvScenario


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))


def _mmr_select(
    query_emb: np.ndarray,
    cand_embs: np.ndarray,
    cand_scores: np.ndarray,
    k: int,
    lambda_mult: float = 0.7,
) -> List[int]:
    """
    Max Marginal Relevance selection.
    cand_scores: similarity to query (higher is better)
    """
    if len(cand_embs) == 0:
        return []
    k = min(k, len(cand_embs))

    selected = []
    # start with max similarity
    first = int(np.argmax(cand_scores))
    selected.append(first)

    while len(selected) < k:
        best_idx = None
        best_val = -1e9
        for i in range(len(cand_embs)):
            if i in selected:
                continue
            sim_q = cand_scores[i]
            sim_s = max(_cos_sim(cand_embs[i], cand_embs[j]) for j in selected)
            val = lambda_mult * sim_q - (1 - lambda_mult) * sim_s
            if val > best_val:
                best_val = val
                best_idx = i
        selected.append(int(best_idx))
    return selected


class DrivingMemory:
    def __init__(self, encode_type='sce_language', db_path=None) -> None:
        self.encode_type = encode_type
        if encode_type != 'sce_language':
            raise ValueError("Only sce_language is supported in this project version.")

        # Embedding backend:
        # - hf: local sentence-transformers (recommended for DeepSeek)
        # - openai/azure: OpenAIEmbeddings
        backend = os.getenv("EMBEDDING_BACKEND")
        if backend is None:
            backend = "hf" if os.getenv("OPENAI_API_BASE", "").find("deepseek") >= 0 else "openai"

        if backend == "hf":
            model_name = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            self.embedding = HuggingFaceEmbeddings(model_name=model_name)
        else:
            if os.environ.get("OPENAI_API_TYPE") == 'azure':
                self.embedding = OpenAIEmbeddings(
                    deployment=os.environ['AZURE_EMBED_DEPLOY_NAME'], chunk_size=1
                )
            else:
                # openai-compatible (OpenAI or DeepSeek if your base supports embeddings)
                self.embedding = OpenAIEmbeddings()

        db_path = os.path.join('./db', 'chroma_5_shot_20_mem/') if db_path is None else db_path
        self.scenario_memory = Chroma(
            embedding_function=self.embedding,
            persist_directory=db_path
        )
        self._db_path = db_path

        print(
            "==========Loaded ", db_path,
            " Memory, Now the database has ",
            len(self.scenario_memory._collection.get(include=['embeddings'])['embeddings']),
            " items.=========="
        )

    def retriveMemory(
        self,
        driving_scenario: EnvScenario,
        frame_id: int,
        top_k: int = 5,
        diverse: bool = False,
        candidate_pool_k: int = 20,
        mmr_lambda: float = 0.7,
    ):
        query_scenario = driving_scenario.describe(frame_id)

        # candidate pool
        pool_k = max(candidate_pool_k, top_k)
        similarity_results = self.scenario_memory.similarity_search_with_score(query_scenario, k=pool_k)

        if not diverse or len(similarity_results) <= top_k:
            return [similarity_results[i][0].metadata for i in range(min(top_k, len(similarity_results)))]

        # MMR selection on embeddings of candidate docs
        cand_docs = [d for (d, s) in similarity_results]
        cand_texts = [d.page_content for d in cand_docs]

        # Query embedding + candidate embeddings (batch)
        query_emb = np.array(self.embedding.embed_query(query_scenario), dtype=np.float32)
        cand_embs = np.array(self.embedding.embed_documents(cand_texts), dtype=np.float32)

        # Similarity to query via cosine (higher better)
        cand_scores = np.array([_cos_sim(query_emb, cand_embs[i]) for i in range(len(cand_embs))], dtype=np.float32)

        selected_idx = _mmr_select(query_emb, cand_embs, cand_scores, k=top_k, lambda_mult=mmr_lambda)
        return [cand_docs[i].metadata for i in selected_idx]

    def addMemory(self, sce_descrip: str, human_question: str, response: str, action: int, sce: EnvScenario = None, comments: str = "", extra_meta: dict = None):
        sce_descrip = sce_descrip.replace("'", "")

        get_results = self.scenario_memory._collection.get(
            where_document={"$contains": sce_descrip}
        )

        meta = {"human_question": human_question, "LLM_response": response, "action": action, "comments": comments}
        if extra_meta:
            meta.update(extra_meta)

        if len(get_results['ids']) > 0:
            _id = get_results['ids'][0]
            self.scenario_memory._collection.update(ids=_id, metadatas=meta)
        else:
            doc = Document(page_content=sce_descrip, metadata=meta)
            self.scenario_memory.add_documents([doc])

    def deleteMemory(self, ids):
        self.scenario_memory._collection.delete(ids=ids)

    def combineMemory(self, other_memory):
        other_documents = other_memory.scenario_memory._collection.get(include=['documents', 'metadatas', 'embeddings'])
        current_documents = self.scenario_memory._collection.get(include=['documents', 'metadatas', 'embeddings'])
        for i in range(len(other_documents['embeddings'])):
            if other_documents['embeddings'][i] in current_documents['embeddings']:
                continue
            self.scenario_memory._collection.add(
                embeddings=other_documents['embeddings'][i],
                metadatas=other_documents['metadatas'][i],
                documents=other_documents['documents'][i],
                ids=other_documents['ids'][i]
            )
