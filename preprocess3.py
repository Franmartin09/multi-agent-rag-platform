import json
import re
from pathlib import Path
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
import os
from google import genai
from pydantic import BaseModel, Field
from typing import List
from google.genai import types
import time

last_gemini_call = None

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
DOCS_DIR = Path("./docs")
OUTPUT_DIR = Path("./output3")

TABLES_DIR = OUTPUT_DIR / "tables"
CHUNKS_DIR = OUTPUT_DIR / "final_chunks"
FIGURES_DIR = OUTPUT_DIR / "rendered_figures"

class DiagramComponent(BaseModel):
    name: str = Field(
        description="Name of the component appearing in the diagram."
    )
    description: str = Field(
        description="Technical description of the component and its role."
    )


class TechnicalDetail(BaseModel):
    detail: str = Field(
        description="Relevant technical information extracted from the diagram."
    )


class WarningItem(BaseModel):
    warning: str = Field(
        description="Warning, note or constraint shown in the diagram."
    )


class DiagramDescription(BaseModel):
    purpose: str = Field(
        description="Purpose of the diagram."
    )

    summary: str = Field(
        description="Overall explanation of what the diagram teaches."
    )

    components: List[DiagramComponent] = Field(
        default_factory=list,
        description="Components, tools, materials or parts appearing in the diagram."
    )

    procedure: List[str] = Field(
        default_factory=list,
        description="Ordered list of steps represented by the diagram."
    )

    technical_details: List[TechnicalDetail] = Field(
        default_factory=list,
        description="Additional relevant technical details."
    )

    warnings: List[WarningItem] = Field(
        default_factory=list,
        description="Warnings, precautions or constraints."
    )

GEMINI_PROMPT="""
You are an expert technical documentation analyst.

Analyze the provided image, which is usually extracted from an engineering or maintenance manual.

The image may contain:
- technical diagrams
- flowcharts
- exploded views
- assembly instructions
- maintenance procedures
- arrows
- labels
- numbered steps
- symbols
- embedded text

Your goal is NOT to simply describe what is visible.

Your goal is to explain what the diagram is trying to teach.

Generate a detailed technical description that could replace the image inside a knowledge base.

Instructions:

1. Identify the purpose of the diagram.
2. Explain the complete procedure or process shown.
3. Integrate all visible text naturally into the explanation.
4. Explain what every important object/component represents.
5. Explain relationships indicated by arrows, connectors or numbering.
6. Describe the sequence of operations in order.
7. Mention tools, materials, seals, pipes, valves, connectors or components when present.
8. Mention warnings, notes or constraints if visible.
9. If dimensions, angles, distances or measurements appear, include them.
10. If multiple subfigures exist (A, B, C...), explain each separately.
11. Ignore purely decorative elements.
12. Do not mention colors unless they have technical meaning.
13. If part of the image is unreadable, explicitly state that instead of guessing.

The output should be written as a coherent technical explanation in Markdown.

Return ONLY structure and valid JSON.

"""

from dotenv import load_dotenv
load_dotenv()

# ==========================================
# CONFIGURACIÓN DE TU GEMINI API KEY
# ==========================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def describe_figure_with_gemini(image_path: Path) -> DiagramDescription:
    """
    Generates a structured semantic description of a technical figure using Gemini.
    """

    try:
        uploaded_file = client.files.upload(file=image_path)

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[
                uploaded_file,
                GEMINI_PROMPT,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DiagramDescription,
            ),
        )

        return response.parsed

    except Exception as e:
        print(f"Gemini error ({image_path.name}): {e}")
        return DiagramDescription(
            purpose="",
            summary="",
            components=[],
            procedure=[],
            technical_details=[],
            warnings=[],
        )

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


def table_to_markdown(table_data: list) -> str:
    """Converts a 2D list (table) into a Markdown formatted string."""
    if not table_data:
        return ""
    
    md_lines = []
    # Obtener el número máximo de columnas para mantener la consistencia
    num_cols = max(len(row) for row in table_data) if table_data else 0
    if num_cols == 0:
        return ""

    for i, row in enumerate(table_data):
        # Rellenar con vacíos si alguna fila tiene menos columnas
        padded_row = row + [""] * (num_cols - len(row))
        
        # Limpiar celdas: los saltos de línea y el símbolo "|" rompen el Markdown
        cleaned_row = [str(cell).replace('\n', ' ').replace('|', '\\|').strip() for cell in padded_row]
        
        md_lines.append("| " + " | ".join(cleaned_row) + " |")
        
        # Añadir la línea separadora de Markdown después de la primera fila (encabezados)
        if i == 0:
            md_lines.append("|" + "|".join(["---"] * num_cols) + "|")
            
    return "\n".join(md_lines)


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
    """Extracts tables as CSV files and returns their metadata, content as markdown, and bounding boxes."""
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
                "bbox": table.bbox,
                "content": table_to_markdown(extracted_table)  # <--- Se guarda como string Markdown
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
        if page_idx!=5:
            # Esperar para respetar el límite de una petición por minuto
            global last_gemini_call
            if last_gemini_call is not None:
                elapsed = time.time() - last_gemini_call

                if elapsed < 60:
                    wait_time = 60 - elapsed
                    print(f"Waiting {wait_time:.1f}s before next Gemini request...")
                    time.sleep(wait_time)

            last_gemini_call = time.time()
            
            # ENVIAR LA IMAGEN A GEMINI PARA EXTRAER EL GEMINI DESCRIPTION
            gemini_description=describe_figure_with_gemini(image_path)
            

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
                "gemini_description": gemini_description.model_dump(),
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