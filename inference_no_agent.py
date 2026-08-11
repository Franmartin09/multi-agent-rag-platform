import os
from dotenv import load_dotenv

import psycopg2
from psycopg2.extras import Json

from google import genai
from google.genai import types

load_dotenv()

# CONFIG
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-3.5-flash"

TOP_K = 3

# CLIENTS
client = genai.Client(api_key=GEMINI_API_KEY)

conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
)

conn.autocommit = True

# EMBEDDING
def embed(text: str):

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        ),
    )

    return response.embeddings[0].values


# RETRIEVAL
def retrieve(question):

    vector = embed(question)

    sql = """
    SELECT
        document_name,
        page_number,
        chunk_type,
        content,
        metadata,
        embedding <=> %s::vector AS distance
    FROM document_chunks
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
    """

    cur = conn.cursor()
    cur.execute(
        sql,
        (
            vector,
            vector,
            TOP_K,
        ),
    )

    rows = cur.fetchall()
    cur.close()

    return rows

# PROMPT
def build_prompt(question, retrieved):

    context = ""

    for i, row in enumerate(retrieved, start=1):

        (
            document,
            page,
            chunk_type,
            content,
            metadata,
            distance,
        ) = row

        context += f"""
==============================
Chunk {i}

Document: {document}
Page: {page}
Type: {chunk_type}
Similarity Distance: {distance:.4f}

{content}

"""

    prompt = f"""
You are a technical assistant.

Answer ONLY using the retrieved documentation.

If the answer is not contained in the documentation, say that you do not have enough information.

Question:

{question}

Retrieved context:

{context}

Provide a detailed answer.
"""

    return prompt

# LLM
def ask_llm(question):

    retrieved = retrieve(question)
    prompt = build_prompt(question, retrieved)

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )

    return response.text, retrieved


# MAIN
def main():

    print()
    print("Technical RAG")
    print("Type exit to quit.")
    print()

    while True:
        question = input(">>> ")

        if question.lower() in ("exit", "quit"):
            break

        answer, docs = ask_llm(question)
        print("\nRetrieved chunks:\n")

        for i, row in enumerate(docs, start=1):
            print(f"{i}. {row[0]} | page {row[1]} | {row[2]} | distance={row[5]:.4f}")

        print("\nAnswer:\n")
        print(answer)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()