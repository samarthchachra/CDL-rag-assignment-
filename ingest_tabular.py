from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import csv
import xlrd


@dataclass
class TabularChunk:
    chunk_id: str
    source_file: str
    source_type: str
    location: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _clean_num(val: Any) -> str:
    s = str(val).strip()
    if s in ["...", "", "NA", "None"]:
        return "0.00"
    try:
        return f"{float(s):.2f}"
    except ValueError:
        return s


def ingest_csv(file_path: Path | str) -> list[TabularChunk]:
    path = Path(file_path)
    chunks: list[TabularChunk] = []

    with open(path, "r", encoding="latin-1") as f:
        reader = csv.reader(f)
        try:
            _ = next(reader)
        except StopIteration:
            return chunks

        for row_idx, row in enumerate(reader, start=2):
            if not row or not any(row):
                continue

            cleaned_row = [c.strip() for c in row]
            if len(cleaned_row) < 9:
                cleaned_row += [""] * (9 - len(cleaned_row))

            (
                cat,
                subcat,
                slno,
                ministry,
                scheme,
                act_21_22,
                be_22_23,
                re_22_23,
                be_23_24,
            ) = cleaned_row[:9]

            if any(k in cat for k in ["Grand Total", "PART A Total", "PART B Total"]):
                chunk_type = "aggregate_total"
                text = (
                    f"Union Gender Budget Statement 2023-24 | Aggregate Row: {cat} | "
                    f"2021-2022 Actuals: Rs {act_21_22} crore | "
                    f"2022-2023 Budget Estimates (BE): Rs {be_22_23} crore | "
                    f"2022-2023 Revised Estimates (RE): Rs {re_22_23} crore | "
                    f"2023-2024 Budget Estimates (BE): Rs {be_23_24} crore. "
                    "All figures in crore rupees."
                )
                loc = f"row with Category '{cat}'"
            elif scheme.lower() == "total":
                chunk_type = "ministry_total"
                text = (
                    f"Union Gender Budget Statement 2023-24 | Category: {cat} | "
                    f"Ministry/Department Total: {ministry} ({subcat}) | "
                    f"2021-2022 Actuals: Rs {act_21_22} crore | "
                    f"2022-2023 Budget Estimates (BE): Rs {be_22_23} crore | "
                    f"2022-2023 Revised Estimates (RE): Rs {re_22_23} crore | "
                    f"2023-2024 Budget Estimates (BE): Rs {be_23_24} crore. "
                    "All figures in crore rupees."
                )
                loc = f"row {row_idx}: {ministry} Total"
            else:
                chunk_type = "scheme"
                text = (
                    f"Union Gender Budget Statement 2023-24 | Category: {cat} | {subcat} | "
                    f"Ministry/Department: {ministry} | Scheme #{slno}: {scheme} | "
                    f"2021-2022 Actuals: Rs {act_21_22} crore | "
                    f"2022-2023 Budget Estimates (BE): Rs {be_22_23} crore | "
                    f"2022-2023 Revised Estimates (RE): Rs {re_22_23} crore | "
                    f"2023-2024 Budget Estimates (BE): Rs {be_23_24} crore. "
                    "All figures in crore rupees."
                )
                loc = f"row {row_idx}: {scheme}"

            chunks.append(
                TabularChunk(
                    chunk_id=f"MRF_row_{row_idx}",
                    source_file=path.name,
                    source_type="csv",
                    location=loc,
                    text=text,
                    metadata={
                        "row": row_idx,
                        "type": chunk_type,
                        "category": cat,
                        "ministry": ministry,
                        "scheme": scheme,
                        "be_2023_24": be_23_24,
                    },
                )
            )

    return chunks


def ingest_xls(file_path: Path | str) -> list[TabularChunk]:
    path = Path(file_path)
    wb = xlrd.open_workbook(str(path))
    sh = wb.sheet_by_name("Sheet1")
    chunks: list[TabularChunk] = []

    current_section = ""
    current_demand = ""
    current_ministry = ""
    curr_scheme_name = ""
    curr_sl = ""

    for r in range(sh.nrows):
        vals = [str(sh.cell(r, c).value).strip() for c in range(sh.ncols)]
        all_str = " ".join(v for v in vals if v)
        if not all_str:
            continue

        if "PART A - 100%" in all_str:
            current_section = "PART A: 100% Women specific programmes"
            continue
        elif "PART B-30%" in all_str or "PART B - 30%" in all_str:
            current_section = "PART B: 30% Women specific programmes"
            continue

        if r in [0, 1, 5, 7, 8, 9, 10, 11, 12]:
            continue

        if r in [2, 3, 4]:
            chunks.append(
                TabularChunk(
                    chunk_id=f"stat20_narrative_{r + 1}",
                    source_file=path.name,
                    source_type="xls",
                    location="narrative note near the top of Sheet1",
                    text=f"Union Budget Statement 20 (2013-14) Narrative Note: {all_str}",
                    metadata={"row": r + 1, "type": "narrative"},
                )
            )
            continue

        c0, c1 = vals[0], vals[1]
        num_vals = vals[2:]
        has_nums = any(x not in ["", "..."] for x in num_vals)

        if "GRAND TOTAL" in all_str.upper():
            chunks.append(
                TabularChunk(
                    chunk_id=f"stat20_grand_total_{r + 1}",
                    source_file=path.name,
                    source_type="xls",
                    location="grand total row of Sheet1",
                    text=(
                        "Union Budget Statement 20 (2013-14) | GRAND TOTAL (PART A and B) | "
                        f"2012-13 Budget Estimates (BE): Total Rs {_clean_num(num_vals[2])} crore "
                        f"(Plan: Rs {_clean_num(num_vals[0])} crore, Non-Plan: Rs {_clean_num(num_vals[1])} crore) | "
                        f"2012-13 Revised Estimates (RE): Total Rs {_clean_num(num_vals[5])} crore "
                        f"(Plan: Rs {_clean_num(num_vals[3])} crore, Non-Plan: Rs {_clean_num(num_vals[4])} crore) | "
                        f"2013-14 Budget Estimates (BE): Total Rs {_clean_num(num_vals[8])} crore "
                        f"(Plan: Rs {_clean_num(num_vals[6])} crore, Non-Plan: Rs {_clean_num(num_vals[7])} crore). "
                        "All figures in crore rupees."
                    ),
                    metadata={"row": r + 1, "type": "grand_total"},
                )
            )
            continue

        if "TOTAL (PART" in all_str.upper() or "TOTAL (PART" in c1.upper():
            next_vals = (
                [str(sh.cell(r + 1, c).value).strip() for c in range(sh.ncols)]
                if r + 1 < sh.nrows
                else []
            )
            nums = num_vals if has_nums else next_vals[2:]
            part_name = "PART A" if "PART A" in all_str.upper() else "PART B"
            chunks.append(
                TabularChunk(
                    chunk_id=f"stat20_part_total_{r + 1}",
                    source_file=path.name,
                    source_type="xls",
                    location=f"Total {part_name} row of Sheet1",
                    text=(
                        f"Union Budget Statement 20 (2013-14) | Total {part_name} | "
                        f"2012-13 Budget Estimates (BE): Total Rs {_clean_num(nums[2])} crore | "
                        f"2012-13 Revised Estimates (RE): Total Rs {_clean_num(nums[5])} crore | "
                        f"2013-14 Budget Estimates (BE): Total Rs {_clean_num(nums[8])} crore. "
                        "All figures in crore rupees."
                    ),
                    metadata={"row": r + 1, "type": "part_total", "part": part_name},
                )
            )
            continue

        if c0.startswith("Demand No"):
            current_demand = c0
            current_ministry = c1 if c1 else ""
            continue
        elif not c0 and c1 and not has_nums and "Total" not in c1:
            if not curr_scheme_name:
                current_ministry = f"{current_ministry} {c1}".strip()
            else:
                curr_scheme_name = f"{curr_scheme_name} {c1}".strip()
            continue

        if c1.strip() == "Total" or c1.strip().startswith("Total "):
            chunks.append(
                TabularChunk(
                    chunk_id=f"stat20_min_total_{r + 1}",
                    source_file=path.name,
                    source_type="xls",
                    location=f"row {r + 1}: {current_ministry} Total",
                    text=(
                        f"Union Budget Statement 20 (2013-14) | {current_section} | "
                        f"{current_demand} - {current_ministry} | Total | "
                        f"2012-13 BE: Rs {_clean_num(num_vals[2])} crore | "
                        f"2012-13 RE: Rs {_clean_num(num_vals[5])} crore | "
                        f"2013-14 BE: Rs {_clean_num(num_vals[8])} crore. "
                        "All figures in crore rupees."
                    ),
                    metadata={
                        "row": r + 1,
                        "type": "ministry_total",
                        "demand": current_demand,
                        "ministry": current_ministry,
                    },
                )
            )
            curr_scheme_name = ""
            curr_sl = ""
            continue

        if c0 and (c0.replace(".", "").isdigit() or c0.isdigit()):
            curr_sl = c0
            curr_scheme_name = c1
        elif c1 and not curr_scheme_name:
            curr_scheme_name = c1
        elif c1 and curr_scheme_name:
            curr_scheme_name = f"{curr_scheme_name} {c1}".strip()

        if has_nums and curr_scheme_name:
            chunks.append(
                TabularChunk(
                    chunk_id=f"stat20_scheme_{r + 1}",
                    source_file=path.name,
                    source_type="xls",
                    location=f"row {r + 1}: {curr_scheme_name}",
                    text=(
                        f"Union Budget Statement 20 (2013-14) | {current_section} | "
                        f"{current_demand} - {current_ministry} | "
                        f"Scheme #{curr_sl}: {curr_scheme_name} | "
                        f"2012-13 BE: Total Rs {_clean_num(num_vals[2])} crore "
                        f"(Plan: Rs {_clean_num(num_vals[0])} crore, Non-Plan: Rs {_clean_num(num_vals[1])} crore) | "
                        f"2012-13 RE: Total Rs {_clean_num(num_vals[5])} crore "
                        f"(Plan: Rs {_clean_num(num_vals[3])} crore, Non-Plan: Rs {_clean_num(num_vals[4])} crore) | "
                        f"2013-14 BE: Total Rs {_clean_num(num_vals[8])} crore "
                        f"(Plan: Rs {_clean_num(num_vals[6])} crore, Non-Plan: Rs {_clean_num(num_vals[7])} crore). "
                        "All figures in crore rupees."
                    ),
                    metadata={
                        "row": r + 1,
                        "type": "scheme",
                        "demand": current_demand,
                        "ministry": current_ministry,
                        "scheme": curr_scheme_name,
                    },
                )
            )
            curr_scheme_name = ""
            curr_sl = ""

    return chunks


def ingest_tabular_data(corpus_dir: Path | str) -> list[TabularChunk]:
    corpus_path = Path(corpus_dir)
    all_chunks: list[TabularChunk] = []

    csv_path = corpus_path / "MRF_13_Union_Budget.csv"
    if csv_path.exists():
        all_chunks.extend(ingest_csv(csv_path))

    xls_path = corpus_path / "stat20.xls"
    if xls_path.exists():
        all_chunks.extend(ingest_xls(xls_path))

    return all_chunks


if __name__ == "__main__":
    import sys

    dir_path = sys.argv[1] if len(sys.argv) > 1 else "corpus"
    items = ingest_tabular_data(dir_path)
    print(f"Total tabular chunks ingested: {len(items)}")
    for sample in items[:3]:
        print(f"[{sample.source_file}] {sample.location} -> {sample.text[:120]}...")
