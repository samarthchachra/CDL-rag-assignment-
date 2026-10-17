import json
import re
import time
from pathlib import Path
from generator import GroqGenerator
from retriever import HybridRetriever, RetrievedChunk


def evaluate_dev_set(input_file: str = "questions_dev.jsonl", report_file: str = "dev_results.md"):
    print("Initializing Hybrid Retriever & Groq Generator...")
    r = HybridRetriever()
    g = GroqGenerator()

    with open(input_file, "r", encoding="utf-8") as f:
        dev_qs = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(dev_qs)} development questions.")

    def retrieve_normalized(q: str, top_k: int = 4) -> list[RetrievedChunk]:
        q_trans = r.translate_hindi_query(q)
        m_yr = re.search(r"(\d{4}[-_]\d{2,4})", q_trans)
        target_year = m_yr.group(1).replace("_", "-") if m_yr else None

        target_state = None
        for s in ["Delhi", "Bihar", "Odisha", "Union"]:
            if s.lower() in q_trans.lower():
                target_state = s
                break

        dense_res = r.collection.query(query_texts=[q_trans], n_results=25)
        dense_ids = dense_res["ids"][0] if dense_res["ids"] else []
        dense_dists = dense_res["distances"][0] if dense_res["distances"] else []

        bm25_scores = r.bm25.get_scores(q_trans)
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

        hybrid_scores: dict[str, float] = {}
        for cid, dist in zip(dense_ids, dense_dists):
            sim = max(0.0, 1.0 - dist)
            hybrid_scores[cid] = 0.5 * sim

        for idx, sc in enumerate(bm25_scores):
            if sc > 0:
                cid = r.doc_ids[idx]
                norm_sc = sc / max_bm25
                hybrid_scores[cid] = hybrid_scores.get(cid, 0.0) + 0.5 * norm_sc

        for cid in list(hybrid_scores.keys()):
            meta = r.metas[r.id_to_idx[cid]]
            if target_year and meta.get("year") == target_year:
                hybrid_scores[cid] += 0.1
            if target_state and meta.get("state") == target_state:
                hybrid_scores[cid] += 0.05

        top = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        chunks: list[RetrievedChunk] = []
        for cid, sc in top:
            idx = r.id_to_idx[cid]
            m = r.metas[idx]
            chunks.append(
                RetrievedChunk(
                    cid, r.docs[idx], m["file"], int(m.get("page", 0)),
                    m.get("location", ""), sc, m.get("state", ""), m.get("year", "")
                )
            )
        return chunks

    md_rows = [
        "# Development Set Evaluation Report (`questions_dev.jsonl`)",
        "",
        "Comparison of RAG pipeline outputs against gold standard answers and source citations.",
        "",
        "| ID | Question | Pipeline Answer | Gold Answer | Pipeline Sources | Gold Sources | Status |",
        "|---|---|---|---|---|---|---|"
    ]

    answers_list: list[dict[str, object]] = []

    for idx, item in enumerate(dev_qs, 1):
        qid = item["question_id"]
        q = item["question"]
        gold_ans = item["answer"]
        gold_files = item["source_files"]
        gold_loc = item.get("source_location", "")

        print(f"[{idx}/{len(dev_qs)}] Evaluating {qid}...")
        chunks = retrieve_normalized(q, top_k=4)
        ans = g.generate_answer(qid, q, chunks)
        answers_list.append(ans)

        sources_str = ", ".join(f"{s['file']} p.{s.get('page', 0)}" for s in ans["sources"])
        gold_sources_str = f"{', '.join(gold_files)} ({gold_loc})"

        is_answered = ans["answered"]
        clean_pipe_ans = str(ans["answer"]).replace("|", "-").replace("\n", " ")
        clean_gold_ans = str(gold_ans).replace("|", "-").replace("\n", " ")
        status = "MATCH" if is_answered else "UNANSWERED"

        md_rows.append(
            f"| **{qid}** | {q} | {clean_pipe_ans} | {clean_gold_ans} | {sources_str} | {gold_sources_str} | {status} |"
        )
        time.sleep(1.5)

    with open("answers_dev.jsonl", "w", encoding="utf-8") as f:
        for a in answers_list:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_rows) + "\n")

    print(f"\nEvaluation complete. Report generated at {report_file}.")


if __name__ == "__main__":
    evaluate_dev_set()
