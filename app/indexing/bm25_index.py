"""BM25 키워드 인덱스"""
import json, re, pickle
from pathlib import Path
from rank_bm25 import BM25Okapi

class BM25Index:
    """
    rank_bm25 기반 키워드 검색 인덱스.
    한국어 형태소 단위 토큰화 (정규식 기반, 간단 버전).
    """
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self._chunk_ids: list[str] = []
        self._chunk_texts: list[str] = []
        self._doc_map: dict[str, str] = {}  # chunk_id -> document_id
        self._bm25: BM25Okapi | None = None
        self._load()
    
    def _tokenize(self, text: str) -> list[str]:
        """한국어 단어 단위 토큰화 (간단 버전)"""
        text = re.sub(r'[^\w\s가-힣a-zA-Z0-9]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 1]
    
    def add_chunks(self, chunk_ids: list[str], texts: list[str], doc_ids: list[str]) -> None:
        for cid, text, doc_id in zip(chunk_ids, texts, doc_ids):
            if cid not in self._chunk_ids:
                self._chunk_ids.append(cid)
                self._chunk_texts.append(text)
                self._doc_map[cid] = doc_id
        self._rebuild()
        self._save()
    
    def search(self, query: str, n_results: int = 50, 
               allowed_doc_ids: list[str] | None = None) -> list[dict]:
        """
        BM25 검색. 반환: [{"chunk_id", "document_id", "score", "text"}]
        """
        if not self._bm25 or not self._chunk_ids:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in indexed:
            if score <= 0:
                break
            cid = self._chunk_ids[idx]
            doc_id = self._doc_map.get(cid, "")
            if allowed_doc_ids and doc_id not in allowed_doc_ids:
                continue
            results.append({
                "chunk_id": cid,
                "document_id": doc_id,
                "score": float(score),
                "text": self._chunk_texts[idx],
            })
            if len(results) >= n_results:
                break
        return results
    
    def _rebuild(self) -> None:
        tokenized = [self._tokenize(t) for t in self._chunk_texts]
        if tokenized:
            self._bm25 = BM25Okapi(tokenized)
    
    def _save(self) -> None:
        with open(self.index_path / "bm25_data.pkl", "wb") as f:
            pickle.dump({
                "chunk_ids": self._chunk_ids,
                "chunk_texts": self._chunk_texts,
                "doc_map": self._doc_map,
            }, f)
    
    def _load(self) -> None:
        pkl_path = self.index_path / "bm25_data.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            self._chunk_ids = data["chunk_ids"]
            self._chunk_texts = data["chunk_texts"]
            self._doc_map = data["doc_map"]
            self._rebuild()
    
    def remove_document(self, document_id: str) -> None:
        new_ids, new_texts, new_map = [], [], {}
        for cid, text in zip(self._chunk_ids, self._chunk_texts):
            if self._doc_map.get(cid) != document_id:
                new_ids.append(cid)
                new_texts.append(text)
                new_map[cid] = self._doc_map[cid]
        self._chunk_ids, self._chunk_texts, self._doc_map = new_ids, new_texts, new_map
        self._rebuild()
        self._save()
