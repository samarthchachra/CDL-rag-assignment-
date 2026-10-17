from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import re
import pymupdf


@dataclass
class DocumentChunk:
    chunk_id: str
    source_file: str
    page: int
    state: str
    year: str
    location: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _extract_year_from_filename(filename: str) -> str:
    match = re.search(r"(\d{4}[-_]\d{2,4})", filename)
    if match:
        return match.group(1).replace("_", "-")
    if "14-Gender_Budget" in filename or "17107782611749468962" in filename:
        return "2024-25"
    return "Unknown"


def parse_delhi_pdf(file_path: Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    doc = pymupdf.open(str(file_path))
    year = _extract_year_from_filename(file_path.name)

    # In 2011-12 and 2012-13 PDFs, the reported budget year is actually 2011-12
    effective_year = "2011-12" if "2011-12" in file_path.name or "2012-13" in file_path.name else year

    for pno, page in enumerate(doc):
        text = page.get_text()
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        current_dept = ""
        i = 0
        while i < len(lines):
            line = lines[i]

            # Detect English department headers
            if any(dept in line for dept in [
                "Education", "Health", "Social Welfare", "Women & Child",
                "Transport", "SC/ST", "Urban", "Tourism"
            ]) and not re.match(r"^\d+$", line) and not any(ord(c) > 127 for c in line):
                current_dept = line
                i += 1
                continue

            # Detect Roman numeral scheme index (e.g., i, ii, iii)
            if re.match(r"^(i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv)\b", line, re.I):
                if i + 1 < len(lines):
                    scheme_name = lines[i + 1]

                    # Filter out corrupted legacy font lines
                    if any(ord(c) > 127 for c in scheme_name) or len(scheme_name) < 3:
                        i += 1
                        continue

                    # Collect numeric values following the scheme title
                    num_idx = i + 2
                    nums: list[str] = []
                    while num_idx < len(lines) and re.match(r"^\d+(\.\d+)?$", lines[num_idx]):
                        nums.append(lines[num_idx])
                        num_idx += 1

                    if nums:
                        re_val = nums[0] if len(nums) > 1 else "0"
                        outlay = nums[1] if len(nums) > 1 else nums[0]
                        wc_flow = nums[2] if len(nums) > 2 else outlay

                        try:
                            cr_val = float(outlay) / 10000.0
                            cr_str = f"Rs {cr_val:.2f} crore"
                        except ValueError:
                            cr_str = "N/A"

                        chunk_text = (
                            f"Delhi Gender Budget Statement {effective_year} | File: {file_path.name} | "
                            f"Page {pno + 1} | Department: {current_dept or 'State Scheme'} | "
                            f"Scheme: {scheme_name} | Revised Estimates: Rs {re_val} thousand | "
                            f"Annual Budget Outlay ({effective_year}): Rs {outlay} thousand ({cr_str}) | "
                            f"Flow to Women's Component: Rs {wc_flow} thousand. "
                            "Units reported in thousands of rupees."
                        )

                        chunks.append(
                            DocumentChunk(
                                chunk_id=f"{file_path.stem}_p{pno + 1}_{len(chunks) + 1}",
                                source_file=file_path.name,
                                page=pno + 1,
                                state="Delhi",
                                year=effective_year,
                                location=f"page {pno + 1}",
                                text=chunk_text,
                                metadata={
                                    "scheme": scheme_name,
                                    "department": current_dept,
                                    "outlay_thousands": outlay,
                                    "outlay_crore": cr_str,
                                    "flow_wc_thousands": wc_flow,
                                },
                            )
                        )
                        i = num_idx
                        continue
            i += 1

    return chunks


def parse_bihar_pdf(file_path: Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    doc = pymupdf.open(str(file_path))

    # Pages 1 to 55 are clean English text; pages 56-142 are corrupted legacy Hindi font
    for pno in range(min(55, len(doc))):
        page = doc[pno]
        text = page.get_text()

        # Split text into paragraphs without dropping bullet-point lists
        paras = [p.strip().replace("\n", " ") for p in text.split("\n\n") if len(p.strip()) > 40]
        for idx, para in enumerate(paras):
            chunk_text = (
                f"Bihar Gender Budget Statement 2024-25 | File: {file_path.name} | "
                f"Page {pno + 1} | {para}"
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=f"bihar_p{pno + 1}_{idx + 1}",
                    source_file=file_path.name,
                    page=pno + 1,
                    state="Bihar",
                    year="2024-25",
                    location=f"page {pno + 1}",
                    text=chunk_text,
                    metadata={"page": pno + 1, "state": "Bihar"},
                )
            )

    return chunks


def parse_odisha_pdf(file_path: Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    doc = pymupdf.open(str(file_path))

    for pno in range(len(doc)):
        page = doc[pno]
        text = page.get_text().strip()
        if len(text) < 40:
            continue

        paras = [p.strip().replace("\n", " ") for p in text.split("\n\n") if len(p.strip()) > 40]
        for idx, para in enumerate(paras):
            chunk_text = (
                f"Odisha Gender Budget Statement 2024-25 | File: {file_path.name} | "
                f"Page {pno + 1} | {para}"
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=f"odisha_p{pno + 1}_{idx + 1}",
                    source_file=file_path.name,
                    page=pno + 1,
                    state="Odisha",
                    year="2024-25",
                    location=f"page {pno + 1}",
                    text=chunk_text,
                    metadata={"page": pno + 1, "state": "Odisha"},
                )
            )

    return chunks


def ingest_all_pdfs(corpus_dir: Path | str) -> list[DocumentChunk]:
    corpus_path = Path(corpus_dir)
    all_chunks: list[DocumentChunk] = []

    for pdf_path in sorted(corpus_path.glob("*.pdf")):
        if "gender_budget_" in pdf_path.name:
            all_chunks.extend(parse_delhi_pdf(pdf_path))
        elif "17107782611749468962" in pdf_path.name:
            all_chunks.extend(parse_bihar_pdf(pdf_path))
        elif "14-Gender_Budget" in pdf_path.name:
            all_chunks.extend(parse_odisha_pdf(pdf_path))

    return all_chunks


if __name__ == "__main__":
    import sys

    dir_path = sys.argv[1] if len(sys.argv) > 1 else "corpus"
    items = ingest_all_pdfs(dir_path)
    print(f"Total PDF chunks ingested: {len(items)}")
    for sample in items[:3]:
        print(f"[{sample.source_file}] p.{sample.page} ({sample.state}) -> {sample.text[:120]}...")
