-- ============================================================
-- Enable pgvector
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- Documents
-- One row per PDF
-- ============================================================

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,

    filename TEXT NOT NULL UNIQUE,

    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- Pages
-- One row per page (main retrieval unit)
-- ============================================================

CREATE TABLE pages (
    id SERIAL PRIMARY KEY,

    document_id INTEGER NOT NULL
        REFERENCES documents(id)
        ON DELETE CASCADE,

    page_number INTEGER NOT NULL,

    free_text TEXT,

    embedding VECTOR(768),

    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(document_id, page_number)
);

-- ============================================================
-- Tables
-- ============================================================

CREATE TABLE tables (
    id SERIAL PRIMARY KEY,

    page_id INTEGER NOT NULL
        REFERENCES pages(id)
        ON DELETE CASCADE,

    table_id TEXT NOT NULL,

    csv_path TEXT NOT NULL,

    bbox JSONB NOT NULL
);

-- ============================================================
-- Figures
-- ============================================================

CREATE TABLE figures (
    id SERIAL PRIMARY KEY,

    page_id INTEGER NOT NULL
        REFERENCES pages(id)
        ON DELETE CASCADE,

    figure_id TEXT NOT NULL,

    image_path TEXT NOT NULL,

    internal_text TEXT,

    bbox JSONB NOT NULL
);

-- ============================================================
-- Useful indexes
-- ============================================================

CREATE INDEX idx_pages_document
ON pages(document_id);

CREATE INDEX idx_tables_page
ON tables(page_id);

CREATE INDEX idx_figures_page
ON figures(page_id);

-- Semantic search index
CREATE INDEX idx_pages_embedding
ON pages
USING hnsw (embedding vector_cosine_ops);