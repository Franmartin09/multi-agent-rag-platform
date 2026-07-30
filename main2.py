import os
from pathlib import Path


from langchain_docling.loader import ExportType
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate



from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ------------------------------------------------------------------
# 1. Configuration & API Key
# ------------------------------------------------------------------
# Set your Gemini API key here (or export GOOGLE_API_KEY in your terminal)
from dotenv import load_dotenv
load_dotenv()

# ==========================================
# CONFIGURACIÓN DE TU GEMINI API KEY
# ==========================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")




# Path to your documents (directory or specific file)
DOCS_DIR = Path("./docs")

# Collect all document paths in /docs (e.g., pdf, docx, html, etc.)
supported_extensions = [".pdf", ".docx", ".pptx", ".html", ".md", ".txt"]
file_paths = [
    str(p) for p in DOCS_DIR.glob("**/*") if p.suffix.lower() in supported_extensions
]

if not file_paths:
    raise FileNotFoundError(f"No valid documents found in {DOCS_DIR.resolve()}")

print(f"Found {len(file_paths)} document(s) to process.")

# ------------------------------------------------------------------
# 2. Document Loading with Docling
# ------------------------------------------------------------------
# Load documents using DoclingLoader (exports as Markdown for layout preservation)
loader = DoclingLoader(
    file_path=file_paths,
    export_type=ExportType.MARKDOWN,
)

documents = loader.load()
print(f"Successfully loaded {len(documents)} document section(s).")

# ------------------------------------------------------------------
# 3. Text Chunking
# ------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(documents)
print(f"Split documents into {len(chunks)} chunks.")

# ------------------------------------------------------------------
# 4. Embeddings & Vector Store Setup
# ------------------------------------------------------------------
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Build the FAISS vector database
vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ------------------------------------------------------------------
# 5. Build the Gemini RAG Chain
# ------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
)

# Custom prompt instructing Gemini to ground answers strictly in context
prompt = ChatPromptTemplate.from_template("""
You are an expert assistant. Answer the user's question based strictly on the provided context below.
If the answer is not contained within the context, state clearly that you do not know.

Context:
{context}

Question:
{input}

Answer:
""")

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# ------------------------------------------------------------------
# 6. Execute Query
# ------------------------------------------------------------------
query = "The pump is not primed, the pump only releases a small flow of water and the pump will not start. How can i solve this?"
print(f"\nAsking: {query}\n" + "-" * 50)

response = rag_chain.invoke({"input": query})

print("\n--- Answer ---")
print(response["answer"])

print("\n--- Sources Used ---")
for i, doc in enumerate(response["context"]):
    source = doc.metadata.get("source", "Unknown file")
    print(f"Source {i+1}: {source}")