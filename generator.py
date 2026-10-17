import json
import os
import re
import time
from typing import Any
from dotenv import load_dotenv
import requests
from retriever import HybridRetriever, RetrievedChunk

load_dotenv()

SYSTEM_PROMPT = """You are an expert budget analyst answering questions strictly based on Indian Gender Budget documents.

GROUND RULES:
1. Answer ONLY using the facts present in the provided context. Do NOT extrapolate, speculate, or use outside knowledge.
2. UNIT PRECISION: Budget figures must always include their exact units (e.g., "Rs 88,044.21 crore", "Rs 20,000 thousand (Rs 2 crore)", "Rs 75,62,291.75 lakh"). If Delhi documents report in thousands, preserve that or note the crore conversion explicitly.
3. MULTIPLE ENTRIES: If a scheme appears under both Part A (100% women-specific) and Part B (30% women-specific), report both figures and specify the Part.
4. OUT-OF-SCOPE / UNANSWERABLE: If the context does not contain sufficient facts to answer the question, or asks for a year, jurisdiction, or metric absent from the corpus (e.g., Kerala, Delhi 2018-19, overall state education budget), you MUST set "answered": false, leave "answer": "", and set "sources": []. Never guess.

RESPONSE FORMAT:
You must reply with valid JSON only:
{
  "answered": true,
  "answer": "<concise answer with unit, or empty string if answered is false>",
  "sources": [
    {"file": "<source_file_name>", "page": <int>}
  ]
}
"""


class GroqGenerator:
    def __init__(self, api_key: str | None = None, model: str = "openai/gpt-oss-120b"):
        raw_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.api_key = raw_key.strip("\"' ")
        self.model = model
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        blocks: list[str] = []
        for i, c in enumerate(chunks, 1):
            page_info = f", Page: {c.page}" if c.page > 0 else f", Location: {c.location}"
            blocks.append(
                f"[Context {i}] (File: {c.source_file}{page_info}, State: {c.state}, Year: {c.year})\n{c.text}"
            )
        return "\n\n".join(blocks)

    def generate_answer(
        self,
        question_id: str,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
        is_in_scope: bool = True
    ) -> dict[str, Any]:
        if not is_in_scope or not retrieved_chunks:
            return {
                "question_id": question_id,
                "answered": False,
                "answer": "",
                "sources": []
            }

        context_str = self.format_context(retrieved_chunks)
        user_prompt = f"QUESTION: {question}\n\nCORPUS CONTEXT:\n{context_str}\n\nProvide the JSON response:"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0
        }

        max_retries = 3
        backoff = 2.0
        for attempt in range(max_retries):
            try:
                resp = requests.post(self.api_url, headers=headers, json=payload, timeout=25)
                if resp.status_code == 429:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                resp.raise_for_status()
                res_json = resp.json()
                content = res_json["choices"][0]["message"]["content"]
                parsed = json.loads(content)

                answered = bool(parsed.get("answered", False))
                answer_text = str(parsed.get("answer", "")).strip() if answered else ""
                raw_sources = parsed.get("sources", [])

                clean_sources: list[dict[str, Any]] = []
                if answered:
                    for s in raw_sources:
                        if isinstance(s, dict) and "file" in s:
                            clean_s: dict[str, Any] = {"file": s["file"]}
                            if "page" in s and int(s["page"]) > 0:
                                clean_s["page"] = int(s["page"])
                            clean_sources.append(clean_s)

                    if not clean_sources and retrieved_chunks:
                        top = retrieved_chunks[0]
                        clean_s = {"file": top.source_file}
                        if top.page > 0:
                            clean_s["page"] = top.page
                        clean_sources.append(clean_s)

                return {
                    "question_id": question_id,
                    "answered": answered,
                    "answer": answer_text,
                    "sources": clean_sources
                }
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"  [Generator Error for {question_id}]: {e}")
                    return {
                        "question_id": question_id,
                        "answered": False,
                        "answer": "",
                        "sources": []
                    }
                time.sleep(backoff)
                backoff *= 2

        return {
            "question_id": question_id,
            "answered": False,
            "answer": "",
            "sources": []
        }
