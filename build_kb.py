from pathlib import Path
from typing import Any
import chromadb
from chromadb.config import Settings

from ingest_pdf import ingest_all_pdfs
from ingest_tabular import ingest_tabular_data


def build_knowledge_base(corpus_dir: str = "corpus", db_path: str = "./chroma_db") -> chromadb.Collection:
    corpus_path = Path(corpus_dir)
    print(f"Loading corpus from {corpus_path.resolve()}...")

    # 1. Ingest tabular data (CSV & XLS)
    tabular_chunks = ingest_tabular_data(corpus_path)
    print(f"Ingested {len(tabular_chunks)} tabular chunks.")

    # 2. Ingest PDF documents (Delhi, Bihar, Odisha)
    pdf_chunks = ingest_all_pdfs(corpus_path)
    print(f"Ingested {len(pdf_chunks)} PDF chunks.")

    # 3. Initialize persistent Chroma client
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )

    # 4. Create or recreate collection
    collection_name = "gender_budget"
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # 5. Prepare batch payloads
    all_ids: list[str] = []
    all_docs: list[str] = []
    all_metadatas: list[dict[str, str | int | float | bool]] = []

    for t_chunk in tabular_chunks:
        all_ids.append(t_chunk.chunk_id)
        all_docs.append(t_chunk.text)
        all_metadatas.append({
            "file": t_chunk.source_file,
            "page": 0,
            "location": t_chunk.location,
            "state": "Union",
            "year": "2023-24" if "MRF" in t_chunk.source_file else "2013-14",
            "source_type": t_chunk.source_type,
        })

    for p_chunk in pdf_chunks:
        all_ids.append(p_chunk.chunk_id)
        all_docs.append(p_chunk.text)
        all_metadatas.append({
            "file": p_chunk.source_file,
            "page": int(p_chunk.page),
            "location": p_chunk.location,
            "state": p_chunk.state,
            "year": p_chunk.year,
            "source_type": "pdf",
        })

    # 6. Insert in batches of 200 to prevent payload limits
    batch_size = 200
    for i in range(0, len(all_ids), batch_size):
        end = i + batch_size
        collection.add(
            ids=all_ids[i:end],
            documents=all_docs[i:end],
            metadatas=all_metadatas[i:end]
        )

    print(f"Successfully indexed {collection.count()} chunks into ChromaDB at {db_path}.")
    return collection


if __name__ == "__main__":
    col = build_knowledge_base()
    sample = col.peek(limit=1)
    print(f"Sample indexed record ID: {sample['ids'][0]}")
    print(f"Sample source file: {sample['metadatas'][0]['file']}")
