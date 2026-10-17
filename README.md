# Indian Gender Budget RAG System

A production-grade, document-aware Retrieval-Augmented Generation (RAG) system built over Indian union and state gender budget documents (Delhi, Bihar, Odisha, Union Budget CSVs, and Ministry statistics).

---

## Deliverables & Core Files

- **`SCOPE.md`**: Formal scope document defining document coverage, ingestion boundaries, and the Hindi query translation architecture.
- **`answers.jsonl`**: Raw pipeline responses for all 18 evaluation questions (`questions_eval.jsonl`) in the required JSONL schema.
- **`NOTES.md`**: Implementation notes detailing our hybrid retrieval architecture, dev benchmark audit, future observability/RAGAS enhancements, and AI tool reflections.
- **`build_kb.py`**: Document parsing and ChromaDB vector indexing script.
- **`run_pipeline.py`**: End-to-end question answering pipeline over JSONL input files.
- **`eval_dev.py`**: Comparative benchmark tool evaluating dev query accuracy against ground truth.

---

## System Architecture

1. **Document Ingestion (`ingest_tabular.py`, `ingest_pdf.py`)**:
   - Tailored row-level chunking for `MRF_13_Union_Budget.csv` (320 chunks) and `stat20.xls` (336 chunks).
   - Document-aware section parsing for Delhi, Bihar, and Odisha PDFs (500 chunks). Legacy 8-bit ASCII Hindi glyphs are excluded in favor of clean English and upfront query translation.
2. **Knowledge Base (`build_kb.py`)**:
   - 1,156 total chunks indexed into ChromaDB vector database using the local `all-MiniLM-L6-v2` dense embedding model.
3. **Hybrid Retrieval (`retriever.py`)**:
   - Normalized hybrid fusion combining Chroma dense semantic search (cosine similarity) with sparse BM25Okapi keyword retrieval.
   - Fiscal year (+0.1) and state identifier (+0.05) dynamic metadata boosts.
   - Two-tier Hindi-to-English query translation layer for bilingual query support.
4. **Structured Generation (`generator.py`)**:
   - Groq API inference (`openai/gpt-oss-120b`) enforcing strict grounding, unit preservation, and JSON output schema (`answered`, `answer`, `sources`). Out-of-scope queries explicitly output `answered: false`.

---

## Setup & Installation

### 1. Clone & Install Dependencies

```bash
git clone <your-repo-url>
cd <repo-name>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

### 3. Build Knowledge Base

Parse corpus documents and index them into ChromaDB:

```bash
python3 build_kb.py
```

### 4. Run the Pipeline & Evaluation

- **Run Dev Benchmark**:
  ```bash
  python3 eval_dev.py
  ```

- **Run Evaluation Questions (generates `answers.jsonl`)**:
  ```bash
  python3 run_pipeline.py questions_eval.jsonl answers.jsonl
  ```
