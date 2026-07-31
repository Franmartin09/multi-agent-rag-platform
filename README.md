# Multi-Agent RAG Platform

A Retrieval-Augmented Generation (RAG) platform for technical documentation.

The project indexes PDF manuals, extracts their textual content, tables, and images, converts them into vector embeddings using Google's Gemini Embedding model, stores them in a PostgreSQL database with pgvector, and allows users to query the documentation through a Gemini LLM.

The repository also includes a LangGraph-based multi-agent workflow implementation.

---

# Project Structure

```text
.
├── db/
│   ├── init.sql
│   ├── postgres_data/
│   └── preprocess/
├── docs/
│   └── *.pdf
├── compose.yml
├── indexer.py
├── inference_no_agent.py
├── agent_workflow_langchain.py
├── requirements.txt
├── README.md
├── DESIGN.md
└── TEST_CASES.md
```

---

# Installation Guide

## 1. Clone the repository

```bash
git clone <repository_url>

cd multi-agent-rag-platform
```

---

## 2. Create a Python virtual environment

Windows

```powershell
python -m venv .venv

.venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure environment variables

Create a `.env` file in the project root.

Example:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_documents
DB_USER=rag
DB_PASSWORD=ragpassword

GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

---

## 5. Start PostgreSQL

Run Docker Compose:

```bash
docker compose up -d
```

Verify the database is running:

```bash
docker ps
```

The container should appear as:

```
rag-postgres
```

---

## 6. Add documentation

Place all PDF manuals inside:

```text
docs/
```

Example:

```text
docs/
    UserManual.pdf
    InstallationGuide.pdf
    Maintenance.pdf
```

---

## 7. Build the Vector Database

Run the indexing pipeline:

```bash
python indexer.py
```

The indexing process will:

* Read every PDF inside the `docs` directory.
* Extract page text.
* Extract tables.
* Extract images.
* Generate one chunk per page.
* Compute Gemini embeddings.
* Store embeddings in PostgreSQL (pgvector).
* Save processed data inside `db/preprocess`.

Depending on the number of PDFs, this process may take several minutes.

---

# User Guide

## Querying the documentation

Run:

```bash
python inference_no_agent.py
```

You will see:

```text
Technical RAG
Type exit to quit.
```

Example:

```text
>>> Why is the pump making noise?
```

The application will:

1. Generate an embedding for the question.
2. Retrieve the most similar documentation pages.
3. Build a context prompt.
4. Ask Gemini to answer using only the retrieved documentation.

Example output:

```text
Retrieved chunks:

1. Pump_Manual.pdf | page 24 | text | distance=0.1324
2. Pump_Manual.pdf | page 25 | table | distance=0.1473
3. Installation.pdf | page 11 | text | distance=0.1810

Answer:

The pump noise is usually caused by...
```

---

## Exiting the application

Type:

```text
exit
```

or

```text
quit
```
---

# Database

The project uses:

* PostgreSQL
* pgvector

The indexed documentation is stored inside the `document_chunks` table.

Each record contains:

* document name
* page number
* chunk type
* extracted content
* metadata
* embedding vector

---

# Generated Data

After indexing, the following data are available:

### PostgreSQL

Vector embeddings stored in:

```text
document_chunks
```

### Local data

Processed artifacts are stored in:

```text
db/preprocess/
```

This directory is intended for development and debugging purposes only. It contains the intermediate preprocessing outputs generated during document indexing, making it easier to inspect and validate the extraction pipeline.

---

# Additional Documentation

For implementation details and architectural decisions, see:

* `DESIGN.md`

For validation examples, see:

* `TEST_CASES.md`

---
# Next Steps

The current implementation provides a complete end-to-end RAG pipeline, but several improvements can be incorporated to make the platform more scalable, maintainable, and production-ready.

## Agent Workflow

Extend the current proof-of-concept by implementing a complete LangGraph/LangChain multi-agent architecture.

Planned agents include:

* **Orchestrator Agent**

  * Coordinates the workflow.
  * Decides which agents should be executed and in which order.

* **Retriever Agent**

  * Reformulates user questions when necessary.
  * Generates multiple retrieval queries.
  * Uses English as the internal retrieval language to improve embedding consistency.
  * Retrieves the most relevant documentation chunks.

* **Answer Agent**

  * Generates the final response exclusively from the retrieved context.
  * Answers in the same language as the user's question.

* **Guardrail Agent**

  * Validates that the generated response is supported by the retrieved documentation.
  * Ensures compliance with company guidelines and assistant behavior.
  * Prevents hallucinations and unsupported claims.

* **Linguistic Agent**

  * Detects the user's language using a lightweight local model.
  * Provides language information to the remaining agents.
  * Reduces LLM calls for language identification.

---

## Frontend

Develop a web interface that allows users to interact naturally with the documentation.

Planned features include:

* Conversational chatbot interface.
* Chat history.
* Display of retrieved document pages.
* Source references and similarity scores.
* Support for multiple conversations.
* A document management section to upload new PDF manuals directly to the knowledge base.
* A document library to browse, review, and manage the documentation currently indexed in the vector database.
* The ability to trigger document indexing from the interface after uploading new files.


---

## Easy Deployment

Package the entire application using Docker to simplify deployment and eliminate dependency and environment inconsistencies.

Future deployment should include:

* PostgreSQL with pgvector.
* Backend services.
* Frontend application.
* Automatic database initialization.
* One-command startup using Docker Compose.