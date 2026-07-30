import json
import re
from pathlib import Path
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
DOCS_DIR = Path("./docs")
OUTPUT_DIR = Path("./output")

TABLES_DIR = OUTPUT_DIR / "tables"
CHUNKS_DIR = OUTPUT_DIR / "smart_chunks"
FIGURES_DIR = OUTPUT_DIR / "rendered_figures"


def create_directories():
    """Creates the output directory structure."""
    for folder in [TABLES_DIR, CHUNKS_DIR, FIGURES_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """Normalizes text by removing unnecessary spaces and blank lines."""
    if not text:
        return ""
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()


def is_inside_any_bbox(word: dict, bboxes: list, margin: float = 2.0) -> bool:
    """Checks whether the center of a word falls inside any bounding box."""
    w_x = (word['x0'] + word['x1']) / 2
    w_y = (word['top'] + word['bottom']) / 2
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox
        if (x0 - margin) <= w_x <= (x1 + margin) and (y0 - margin) <= w_y <= (y1 + margin):
            return True
    return False


def process_tables_in_page(page, pdf_name: str, page_idx: int) -> tuple[list, list]:
    """Extracts tables as CSV files and returns their metadata and bounding boxes."""
    tables_meta = []
    table_bboxes = []
    tables = page.find_tables()

    for t_idx, table in enumerate(tables, start=1):
        extracted_table = []

        for row in table.rows:
            row_data = []

            for cell_bbox in row.cells:
                if cell_bbox is None:
                    row_data.append("")
                    continue

                x0, top, x1, bottom = cell_bbox
                cell_crop = page.crop(cell_bbox)
                text = clean_text(cell_crop.extract_text() or "")

                # Replace non-alphanumeric symbols with "X"
                if text and not re.search(r'[a-zA-Z0-9]', text):
                    text = "X"

                # If the cell is empty, check for graphical content
                if not text:
                    margin = 2

                    if (x1 - x0) > (2 * margin) and (bottom - top) > (2 * margin):
                        inner_bbox = (x0 + margin, top + margin, x1 - margin, bottom - margin)
                    else:
                        inner_bbox = cell_bbox

                    inner_crop = page.crop(inner_bbox)

                    has_graphics = (
                        len(inner_crop.curves) > 0 or
                        len(inner_crop.lines) > 0 or
                        len(inner_crop.rects) > 0 or
                        len(inner_crop.images) > 0
                    )

                    if has_graphics:
                        text = "X"

                row_data.append(text)

            extracted_table.append(row_data)

        if extracted_table:
            df = pd.DataFrame(extracted_table)

            table_file = TABLES_DIR / f"{pdf_name}_p{page_idx}_table_{t_idx}.csv"
            df.to_csv(table_file, index=False, header=False, encoding="utf-8-sig")

            tables_meta.append({
                "table_id": f"table_{t_idx}",
                "csv_file": str(table_file),
                "bbox": table.bbox
            })

            table_bboxes.append(table.bbox)

    return tables_meta, table_bboxes


def process_figures_in_page(page, fitz_page, words: list, pdf_name: str, page_idx: int) -> tuple[list, list]:
    """Detects significant visual blocks (diagrams) while ignoring small icons."""
    figures_meta = []
    figure_bboxes = []
    page_area = page.width * page.height

    candidate_bboxes = []

    # 1. Detect large rectangular frames
    for r in page.rects:
        rect_area = r['width'] * r['height']
        if 0.10 * page_area <= rect_area <= 0.85 * page_area:
            candidate_bboxes.append((r['x0'], r['top'], r['x1'], r['bottom']))

    # 2. If no frames are found, search for figure references with complex graphics
    if not candidate_bboxes:
        fig_words = [
            w for w in words
            if re.search(r'\bFig(?:ure|\.|\b)', w['text'], re.IGNORECASE)
            and not w['text'].startswith('(')
            and not w['text'].endswith(')')
        ]

        if fig_words:
            fig_words.sort(key=lambda x: x['top'])
            prev_top = 0

            for marker in fig_words:
                bottom_boundary = marker['bottom'] + 10
                test_bbox = (0, prev_top, page.width, bottom_boundary)
                crop_zone = page.crop(test_bbox)

                # Ignore small icons
                big_rects = [r for r in crop_zone.rects if r['width'] > 40 and r['height'] > 40]
                big_images = [img for img in crop_zone.images if img['width'] > 50 and img['height'] > 50]
                complex_vectors = (len(crop_zone.curves) + len(crop_zone.lines)) > 8

                if big_rects or big_images or complex_vectors:
                    words_in_zone = [
                        w for w in words
                        if test_bbox[1] <= w['top'] <= test_bbox[3]
                    ]

                    if len(words_in_zone) < 35:
                        candidate_bboxes.append(test_bbox)
                        prev_top = bottom_boundary

    # Render and save detected figures
    for idx, bbox in enumerate(candidate_bboxes, start=1):
        fig_id = f"figure_block_{idx}"

        rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
        pix = fitz_page.get_pixmap(clip=rect, dpi=200)

        image_path = FIGURES_DIR / f"{pdf_name}_p{page_idx}_{fig_id}.png"
        pix.save(image_path)

        # Extract internal text from the figure
        fig_words = [
            w['text']
            for w in words
            if bbox[1] - 5 <= w['top'] and w['bottom'] <= bbox[3] + 5
        ]

        clean_fig_text = clean_text(" ".join(fig_words))

        figures_meta.append({
            "figure_id": fig_id,
            "png_file": str(image_path),
            "internal_text": clean_fig_text,
            "bbox": bbox
        })

        figure_bboxes.append(bbox)

    return figures_meta, figure_bboxes


def process_pdf(pdf_path: Path):
    """Processes a document in a single unified pass."""
    pdf_name = pdf_path.stem

    print("\n==========================================")
    print(f"PROCESSING DOCUMENT (NO REDUNDANCY): {pdf_path.name}")
    print("==========================================")

    doc_fitz = fitz.open(pdf_path)

    with pdfplumber.open(pdf_path) as pdf_plumber:
        for page_idx, page in enumerate(pdf_plumber.pages, start=1):
            fitz_page = doc_fitz[page_idx - 1]
            words = page.extract_words()

            # 1. Extract tables
            tables_meta, table_bboxes = process_tables_in_page(page, pdf_name, page_idx)

            # 2. Extract figures
            figures_meta, figure_bboxes = process_figures_in_page(
                page, fitz_page, words, pdf_name, page_idx
            )

            # 3. Extract text excluding tables and figures
            excluded_bboxes = table_bboxes + figure_bboxes

            def filter_outside_bboxes(obj):
                """Filters out characters and graphical elements inside excluded bounding boxes."""
                if "x0" not in obj or "top" not in obj:
                    return True

                w_x = (obj['x0'] + obj['x1']) / 2
                w_y = (obj['top'] + obj['bottom']) / 2
                margin = 2.0

                for bbox in excluded_bboxes:
                    x0, y0, x1, y1 = bbox

                    if (
                        (x0 - margin) <= w_x <= (x1 + margin)
                        and (y0 - margin) <= w_y <= (y1 + margin)
                    ):
                        return False

                return True

            filtered_page = page.filter(filter_outside_bboxes)
            raw_text = filtered_page.extract_text(layout=False) or ""
            free_text = clean_text(raw_text)

            # 4. Save unified JSON chunk only if the page contains content
            if free_text or tables_meta or figures_meta:
                page_chunk = {
                    "document": pdf_path.name,
                    "page": page_idx,
                    "free_text": free_text,
                    "tables": tables_meta,
                    "figures": figures_meta
                }

                json_file = CHUNKS_DIR / f"{pdf_name}_p{page_idx}_chunk.json"

                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(page_chunk, f, ensure_ascii=False, indent=4)

                print(
                    f"  [Page {page_idx}] -> "
                    f"Text: {len(free_text) > 0} | "
                    f"Tables: {len(tables_meta)} | "
                    f"Figures: {len(figures_meta)}"
                )


def main():
    create_directories()

    pdf_files = list(DOCS_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{DOCS_DIR.resolve()}'")
        return

    for pdf_file in pdf_files:
        process_pdf(pdf_file)

    print(f"\nProcessing completed successfully. Check '{OUTPUT_DIR.resolve()}'")


if __name__ == "__main__":
    main()