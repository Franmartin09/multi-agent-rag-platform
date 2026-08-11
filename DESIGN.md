# Design Decisions

This document describes the main design decisions taken during the implementation of the indexing pipeline and explains the reasoning behind them.

## Document Indexing Pipeline

Before storing the documentation in the vector database, every PDF is processed by a dedicated indexing script. The objective of this script is to transform engineering manuals into structured information that can later be retrieved efficiently by the RAG pipeline.

During this process, every PDF page is analyzed to extract three different types of information:

- Plain text
- Tables
- Technical figures and diagrams

Each content type requires a different processing strategy.

---

## Text Extraction

The textual content of each page is extracted while excluding the regions occupied by tables and figures.

This avoids duplicated information and produces a clean textual representation of the document.

---

## Table Processing

Although tables are detected independently during parsing, they are **not stored as isolated chunks**.

Instead, every table is converted into **Markdown format** and appended to the textual content of the same page.

This decision was made because, in technical manuals, tables rarely have meaning on their own. Their interpretation usually depends on the surrounding paragraphs that introduce or explain them.

For example, a troubleshooting table normally follows a sentence such as:

> The following table lists the possible causes and solutions.

If the table were embedded independently, semantic retrieval could lose this important context.

Markdown was selected because it preserves the tabular structure while remaining plain text, allowing embedding models to capture both the content and the relationships between rows and columns.

---

## Page-Level Chunking

The indexing pipeline generates **one text chunk per page** instead of using fixed-size character chunks.

Technical manuals are naturally organized by pages, where procedures, maintenance instructions, troubleshooting guides and diagrams are usually self-contained.

Using page-based chunks provides several advantages:

- Keeps related information together.
- Preserves the original document context.
- Prevents splitting procedures across multiple chunks.
- Simplifies traceability back to the original document.

This strategy also avoids separating explanatory paragraphs from the tables that belong to them.

---

## Technical Figures

Engineering manuals contain a large number of diagrams, exploded views, installation drawings and maintenance illustrations.

These images often contain essential information that cannot be recovered using OCR alone, since their meaning depends on arrows, symbols, layouts and the relationship between graphical elements.

To preserve this information, every relevant figure is extracted as an image and processed using the Gemini API.

Instead of generating a simple caption, Gemini is prompted to produce a detailed structured description that includes information such as:

- Purpose of the figure
- Procedure being illustrated
- Components involved
- Technical details
- Warnings
- Relevant embedded text

The objective is to transform the visual information into semantic text that can be embedded and retrieved by the vector database.

This allows users to ask questions about procedures illustrated only in diagrams without requiring the original image during retrieval.

---

## Unified Representation

At the end of the indexing process, each page is represented by:

- A text chunk containing:
  - Free text
  - Tables converted to Markdown
- One semantic description for every technical figure found on the page.

This approach keeps the contextual relationship between textual information and tables while also making graphical information searchable through semantic retrieval.


## PostgreSQL + pgvector

The extracted information is stored in PostgreSQL using the **pgvector** extension instead of a dedicated vector database.

This decision was made for several reasons.

### Single Storage System

Using PostgreSQL allows both structured metadata and vector embeddings to be stored in the same database.

Each indexed element contains information such as:

- Document name
- Page number
- Chunk type
- Original content
- Metadata
- Vector embedding

Keeping all of this together simplifies indexing, retrieval and future maintenance without introducing an additional infrastructure component.

---

### Separation by Content Type

Although text and figures originate from the same page, they are stored as independent records.

Each record is identified by its `chunk_type`, which currently can be:

- `text`
- `figure`

This design allows semantic searches to retrieve either textual explanations or figure descriptions independently, depending on which embedding is more relevant to the user's query.

For example, a question about a maintenance procedure may retrieve the textual explanation, while a question regarding an installation diagram may retrieve the semantic description generated from the corresponding figure.

---

### Metadata

Each chunk stores metadata alongside its embedding.

Typical metadata includes:

- Source document
- Page number
- Bounding boxes (for figures)
- Image path
- Additional information generated during indexing

This metadata is not embedded, but is useful during retrieval to provide traceability and to reconstruct the original source of the information.

---

### Semantic Search

The content itself is embedded using the **Gemini Embedding API** before being inserted into PostgreSQL.

The resulting embedding vector is stored in a `vector` column provided by pgvector.

When a user submits a query, the same embedding model is used to generate an embedding for the question. PostgreSQL then performs a similarity search over the stored vectors to retrieve the most relevant chunks.

This approach enables semantic retrieval instead of relying on keyword matching, allowing the system to find relevant information even when the user's wording differs from the original documentation.

---

### Why PostgreSQL Instead of a Dedicated Vector Database?

For this project, PostgreSQL with pgvector provides all the required functionality while keeping the architecture simple.

The main advantages are:

- Mature and reliable relational database.
- Native support for vector similarity search through pgvector.
- Structured metadata and embeddings stored together.
- Easy integration with LangChain.
- No need to deploy or maintain an additional vector database.

Since the project indexes technical manuals rather than billions of documents, PostgreSQL offers an excellent balance between simplicity, maintainability and performance.