# Implementation Notes & Architecture

## 1. Document Ingestion & Knowledge Base Construction
- **Corpus Coverage**: Ingested all 11 corpus files across PDF, CSV, and XLS formats using tailored extraction pipelines:
  - *Tabular (CSV/XLS)*: Converted row-level expenditure items from the Union Budget (`MRF_13_Union_Budget.csv`) and Ministry statistics (`stat20.xls`) into self-contained, natural language factual chunks with explicit column-row context.
  - *PDFs*: Structured document parsing across 7 Delhi budget statements, Bihar, and Odisha gender budgets. Extracted clean English sections while excluding legacy 8-bit ASCII Kruti Dev Hindi glyphs that lacked search utility.
- **Chunking Strategy**: Applied document-aware semantic chunking (scheme-level blocks for PDFs, row-level statements for tabular data) rather than arbitrary fixed-token windowing to preserve numerical integrity and column bindings.
- **Knowledge Base**: Indexed 1,156 chunks into ChromaDB vector database using the local `all-MiniLM-L6-v2` dense embedding model.

## 2. Hybrid Retrieval & Generation Pipeline
- **Hybrid Retrieval**: Combines dense semantic search (cosine similarity via ChromaDB) with sparse keyword retrieval (BM25Okapi) using min-max score normalization. Includes domain-specific metadata boosts for target fiscal years and state identifiers, plus a query pre-translation layer (Hindi to English) for bilingual query support.
- **Structured Generation**: Utilizes `openai/gpt-oss-120b` via the Groq API. Prompts enforce strict source grounding, numerical exactness, unit extraction (e.g., Crores, Lakhs), and formatted JSON outputs (`answered`, `answer`, `sources`). Unanswerable or out-of-scope queries explicitly output `answered: false`.

## 3. Evaluation & Validation Workflow
- **Development Benchmarking**: Executed the pipeline against all 12 development queries in `questions_dev.jsonl` using a dedicated validation script (`eval_dev.py`). Outputs were manually audited against ground-truth figures for numerical accuracy, unit fidelity, and correct out-of-scope rejection (achieving 100% / 12/12 agreement).
- **Evaluation Set Run**: With dev validation verified, executed the complete pipeline on the 18 evaluation questions (`questions_eval.jsonl`), saving raw structured outputs to `answers.jsonl`.

## 4. What We Would Improve With More Time
- **Golden Dataset Expansion**: Scale the benchmark from 30 total queries to a comprehensive 500+ query golden test set covering complex cross-table arithmetic, comparative multi-year trends, and adversarial out-of-scope prompts.
- **Automated RAG Evaluation Frameworks**: Integrate **RAGAS** or TruLens to track standard quantitative metrics:
  - *Faithfulness*: Measuring whether every factual claim in the generated answer is strictly grounded in retrieved chunks.
  - *Answer Relevance*: Assessing whether the response directly addresses the query without extraneous hallucinated commentary.
  - *Context Precision & Context Recall*: Measuring the rank quality and completeness of retrieved context chunks.
- **Observability & Tracing**: Instrument OpenTelemetry (or Arize Phoenix / LangSmith) to trace per-step retriever latency, embedding drift, token consumption, and automated failure categorization.
- **Advanced Ingestion**: Implement Kruti Dev legacy font transcoding to recover Hindi budget tables, alongside a Text-to-SQL / Code Interpreter engine for complex aggregate calculations.

## 5. AI Tool Reflection
- The AI tool initially attempted to proceed with the analysis and ingestion of the Hindi text directly from the documents. However, it hit a critical roadblock because the PDFs used legacy 8-bit ASCII font encodings (Kruti Dev) that rendered as unindexed garbled characters, making standard tokenization and dense embedding fail. The AI persisted in attempting to process and retain the corrupt text rather than discarding it. I stepped in and made the architectural decision against the AI's approach to explicitly strip the legacy Hindi content from document ingestion, relying on the clean English sections and introducing an upfront query translation layer instead.
