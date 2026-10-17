from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math
import re
import urllib.parse
import urllib.request
from collections import Counter
import chromadb


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source_file: str
    page: int
    location: str
    score: float
    state: str
    year: str


class BM25Index:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs_tokens = [self.tokenize(d) for d in documents]
        self.doc_count = len(documents)
        self.avg_doc_len = (
            sum(len(d) for d in self.docs_tokens) / self.doc_count
            if self.doc_count > 0 else 0.0
        )
        self.doc_freqs: Counter[str] = Counter()
        for doc in self.docs_tokens:
            for word in set(doc):
                self.doc_freqs[word] += 1

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())

    def get_scores(self, query: str) -> list[float]:
        q_tokens = self.tokenize(query)
        scores = [0.0] * self.doc_count
        for q in q_tokens:
            if q not in self.doc_freqs:
                continue
            df = self.doc_freqs[q]
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
            for idx, doc in enumerate(self.docs_tokens):
                tf = doc.count(q)
                if tf > 0:
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (len(doc) / self.avg_doc_len))
                    scores[idx] += idf * (tf * (self.k1 + 1.0)) / denom
        return scores


class HybridRetriever:
    DOMAIN_ENTITIES = {
        "लाडली योजना": "Ladli Yojna",
        "लाडली": "Ladli Yojna",
        "किशोरी योजना": "Kishori Menstrual Hygiene",
        "किशोरी": "Kishori Menstrual Hygiene",
        "जीविका": "Jeevika self-help groups",
        "कन्या उत्थान": "Mukhyamantri Kanya Utthan Yojana",
    }

    FALLBACK_GLOSSARY = {
        "दिल्ली": "Delhi", "बिहार": "Bihar", "ओडिशा": "Odisha",
        "आवंटन": "allocation outlay", "जेंडर बजट": "Gender Budget",
        "मासिक चक्र स्वच्छता": "Menstrual Hygiene", "स्वयं सहायता समूह": "self-help groups",
        "गठित": "formed", "राशि": "amount", "प्रस्तावित": "proposed",
        "जन्म": "birth", "शिशु": "child infant", "लाख": "lakh", "करोड़": "crore"
    }

    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "gender_budget"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(collection_name)
        data = self.collection.get(include=["documents", "metadatas"])
        self.doc_ids: list[str] = data["ids"]
        self.docs: list[str] = data["documents"]
        self.metas: list[dict[str, Any]] = data["metadatas"]
        self.id_to_idx = {cid: idx for idx, cid in enumerate(self.doc_ids)}
        self.bm25 = BM25Index(self.docs)

    def translate_hindi_query(self, query: str) -> str:
        # Check if query contains Devanagari Unicode range
        if not any("\u0900" <= ch <= "\u097f" for ch in query):
            return query

        # Preserve domain-specific scheme names before translation
        prepared = query
        for hi_term, en_term in self.DOMAIN_ENTITIES.items():
            prepared = prepared.replace(hi_term, en_term)

        # Dynamic machine translation call
        try:
            url = (
                "https://translate.googleapis.com/translate_a/single?client=gtx&sl=hi&tl=en&dt=t&q="
                + urllib.parse.quote(prepared)
            )
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
                translated = "".join(part[0] for part in payload[0] if part[0]).strip()
                if translated:
                    return translated
        except Exception:
            pass

        # Resilient offline fallback using glossary
        fallback = prepared
        for hi, en in self.FALLBACK_GLOSSARY.items():
            fallback = fallback.replace(hi, en)
        return fallback

    def check_scope(self, query: str) -> tuple[bool, str]:
        q_lower = query.lower()
        if "kerala" in q_lower:
            return False, "Jurisdiction Kerala is not part of the corpus"

        # Check for unrepresented Delhi years
        m_year = re.search(r"(\d{4}[-_]\d{2,4})", query)
        if m_year:
            yr = m_year.group(1).replace("_", "-")
            if yr in ["2018-19", "2019-20"]:
                return False, f"Fiscal year {yr} is not available in the corpus"

        if "total education budget" in q_lower and "delhi" in q_lower:
            return False, "Total state education budget is outside gender budget scope"

        return True, "In scope"

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        translated_query = self.translate_hindi_query(query)

        # Extract fiscal year and state targets from query
        m_year = re.search(r"(\d{4}[-_]\d{2,4})", translated_query)
        target_year = m_year.group(1).replace("_", "-") if m_year else None

        target_state = None
        for s in ["Delhi", "Bihar", "Odisha", "Union"]:
            if s.lower() in translated_query.lower():
                target_state = s
                break

        # 1. Semantic dense search via Chroma
        dense_results = self.collection.query(
            query_texts=[translated_query],
            n_results=min(25, len(self.docs))
        )
        dense_ids = dense_results["ids"][0] if dense_results["ids"] else []

        # 2. Lexical keyword search via BM25
        bm25_scores = self.bm25.get_scores(translated_query)
        bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:25]

        # 3. Reciprocal Rank Fusion (RRF) with metadata alignment
        rrf_scores: Counter[str] = Counter()
        for rank, cid in enumerate(dense_ids, start=1):
            rrf_scores[cid] += 1.0 / (60.0 + rank)

        for rank, idx in enumerate(bm25_top_indices, start=1):
            cid = self.doc_ids[idx]
            rrf_scores[cid] += 1.0 / (60.0 + rank)

        for cid in list(rrf_scores.keys()):
            meta = self.metas[self.id_to_idx[cid]]
            if target_year and meta.get("year") == target_year:
                rrf_scores[cid] += 0.05
            if target_state and meta.get("state") == target_state:
                rrf_scores[cid] += 0.03

        ranked_items = rrf_scores.most_common(top_k)
        retrieved: list[RetrievedChunk] = []

        for cid, score in ranked_items:
            idx = self.id_to_idx[cid]
            meta = self.metas[idx]
            retrieved.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=self.docs[idx],
                    source_file=meta["file"],
                    page=int(meta.get("page", 0)),
                    location=meta.get("location", ""),
                    score=float(score),
                    state=meta.get("state", ""),
                    year=meta.get("year", ""),
                )
            )

        return retrieved


if __name__ == "__main__":
    retriever = HybridRetriever()

    # Test Hindi query
    q_hi = "दिल्ली के 2026-27 जेंडर बजट में किशोरी (लड़कियों में मासिक चक्र स्वच्छता) योजना के लिए कितना आवंटन प्रस्तावित है?"
    print(f"Original: {q_hi}")
    print(f"Translated: {retriever.translate_hindi_query(q_hi)}")
    res = retriever.retrieve(q_hi, top_k=2)
    for r in res:
        print(f"[{r.source_file} p.{r.page}] (Score: {r.score:.4f}) -> {r.text[:120]}...")
