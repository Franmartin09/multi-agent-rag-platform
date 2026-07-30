import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from pathlib import Path
import json
from dotenv import load_dotenv
load_dotenv()

# ==========================================
# CONFIGURACIÓN DE TU GEMINI API KEY
# ==========================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

OUTPUT_DIR = Path("./output")
CHUNKS_DIR = OUTPUT_DIR / "smart_chunks"

COLLECTION_NAME = "technical_manuals"

DATABASE_URL = (
    "postgresql+psycopg://postgres:postgres@localhost:5432/rag"
)


# ==========================================================
# Vector Store
# ==========================================================
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=DATABASE_URL,
    use_jsonb=True,
)

# ==========================================================
# Helpers
# ==========================================================
def load_chunk(chunk_file: Path) -> Document:
    """
    Converts one JSON chunk into one LangChain Document.
    """

    with open(chunk_file, "r", encoding="utf8") as f:
        data = json.load(f)

    page_content = []

    # -----------------------
    # Main text
    # -----------------------

    if data["free_text"]:

        page_content.append(
            f"TEXT\n{data['free_text']}"
        )

    # -----------------------
    # Table references
    # -----------------------

    if data["tables"]:

        page_content.append("\nTABLES\n")

        for table in data["tables"]:

            page_content.append(
                f"{table['table_id']}"
            )

    # -----------------------
    # Figure references
    # -----------------------

    if data["figures"]:

        page_content.append("\nFIGURES\n")

        for figure in data["figures"]:

            page_content.append(
                figure["internal_text"]
            )

    metadata = {

        "document": data["document"],

        "page": data["page"],

        "tables": data["tables"],

        "figures": data["figures"]

    }

    return Document(

        page_content="\n".join(page_content),

        metadata=metadata

    )


# ==========================================================
# Main
# ==========================================================


def main():

    json_files = sorted(CHUNKS_DIR.glob("*.json"))

    if not json_files:
        print("No chunks found.")
        return

    documents = []

    for file in json_files:

        doc = load_chunk(file)

        documents.append(doc)

    print(f"{len(documents)} documents loaded.")

    vector_store.add_documents(documents)

    print("Indexing completed.")


if __name__ == "__main__":
    main()