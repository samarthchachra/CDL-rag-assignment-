# Scope of the RAG System

### 1. What Was Included
All 11 files in the provided corpus (100% coverage across 9 PDFs, 1 CSV, and 1 XLS) were included without skipping any document. Because government budget files vary widely in formatting, a single generic parser was insufficient. Each document family was ingested through a dedicated processing pipeline:
- **CSV (`MRF_13_Union_Budget.csv`)**: Row-level serialization parsing Part A, Part B, and aggregate total rows with explicit crore units.
- **XLS (`stat20.xls`)**: Dual-mode extraction capturing both top narrative context (introductory policy notes) and multi-row merged tabular rows.
- **Delhi State Statements (7 PDFs)**: Scheme-level table extraction across fiscal years (2011–12 to 2026–27), with thousands-to-crore unit conversion and strict fiscal year metadata stamping.
- **Bihar & Odisha Statements**: Paragraph-level semantic chunking over policy chapters,tables, and budgets.

---

### 2. What Was Excluded & Multilingual Strategy
- **Hindi Text Exclusion:** Multiple documents were bilingual (English and Hindi). Detailed inspection revealed that the Hindi sections were parallel translations of the accompanying English content, and the Hindi text in the PDFs was encoded in legacy 8-bit ASCII fonts (Kruti Dev), which extracted as corrupted character noise. Retaining this noise would degrade embedding and lexical indexing. The legacy Hindi text was discarded, retaining only the clean English text where all official scheme titles, numbers, and units reside.
- **Why the Hindi text exclusion was done and the query translation architecture was used:** Because the assignment evaluation dataset and expected answers are in English, indexing corrupted Hindi text introduced severe vocabulary noise without adding any missing factual information. By translating incoming Hindi queries into English prior to retrieval, the system achieves cross-lingual alignment directly against the clean English index, ensuring high-accuracy lexical (BM25) and dense retrieval across all queries.

---

### 3. What Was Handled Imperfectly & Why
- **Borderless PDF Tables:** Delhi state budget documents feature borderless tabular data where column lines are visual rather than structural. Naive line by line parsing risked interleaving column headers with scheme rows across page breaks, requiring heuristic line-sequence reconstruction rather than deterministic cell parsing.
