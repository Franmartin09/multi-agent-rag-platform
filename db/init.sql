CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_name TEXT NOT NULL,
    page_number INT NOT NULL,
    chunk_type TEXT NOT NULL, -- Puede ser: 'text', 'table', 'figure'
    content TEXT NOT NULL,    -- El texto exacto que el modelo de embeddings vectorizará y el LLM leerá
    embedding vector(3072),   -- El vector (ajusta el 1536 a la dimensión de tu modelo, ej. OpenAI = 1536)
    metadata JSONB            -- Para guardar rutas de imágenes, bounding boxes, JSON original, etc.
);

-- Índice para acelerar la búsqueda vectorial
CREATE INDEX ON document_chunks USING hnsw (embedding vector_cosine_ops);