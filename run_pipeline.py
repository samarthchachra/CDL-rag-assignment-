import json
import re
import sys
import time
from pathlib import Path
from generator import GroqGenerator
from retriever import HybridRetriever, RetrievedChunk


def run_pipeline(input_file: str = "questions_eval.jsonl", output_file: str = "answers.jsonl") -> list[dict[str, object]]:
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: {input_file} not found.")
        sys.exit(1)

    print(f"Initializing Hybrid Retriever & Groq Generator...")
    r = HybridRetriever()
    g = GroqGenerator()

    with open(input_path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(questions)} evaluation questions from {input_file}.")

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

    results: list[dict[str, object]] = []

    start_time = time.time()
    for idx, item in enumerate(questions, 1):
        qid = item["question_id"]
        q_text = item["question"]
        print(f"\n[{idx}/{len(questions)}] Processing {qid}: {q_text[:70]}...")

        # 1. Scope Check
        in_scope, scope_reason = r.check_scope(q_text)
        if not in_scope:
            print(f"  -> Out-of-scope ({scope_reason}). Setting answered=False.")
            out: dict[str, object] = {
                "question_id": qid,
                "answered": False,
                "answer": "",
                "sources": []
            }
            results.append(out)
            continue

        # 2. Normalized Hybrid Retrieval
        chunks = retrieve_normalized(q_text, top_k=4)
        top_meta = f"{chunks[0].source_file} p.{chunks[0].page}" if chunks else "None"
        print(f"  -> Retrieved top chunk: {top_meta}")

        # 3. Generation via Groq
        answer_obj = g.generate_answer(
            question_id=qid,
            question=q_text,
            retrieved_chunks=chunks,
            is_in_scope=in_scope
        )
        print(f"  -> Answered: {answer_obj['answered']} | {answer_obj['answer'][:80]}")
        results.append(answer_obj)
        time.sleep(2.0)  # Safe rate-limiting throttle

    elapsed = time.time() - start_time
    print(f"\nFinished in {elapsed:.2f} seconds ({elapsed / len(questions):.2f}s per question).")

    if output_file:
        out_path = Path(output_file)
        with open(out_path, "w", encoding="utf-8") as out_f:
            for r_item in results:
                out_f.write(json.dumps(r_item, ensure_ascii=False) + "\n")
        print(f"Results written to {output_file}")

    return results


if __name__ == "__main__":
    target_input = sys.argv[1] if len(sys.argv) > 1 else "questions_eval.jsonl"
    target_output = sys.argv[2] if len(sys.argv) > 2 else "answers.jsonl"
    run_pipeline(target_input, target_output)
