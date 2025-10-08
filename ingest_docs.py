import os
import sys
from google.genai import types
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()
from core.config import settings

embeddings = GoogleGenerativeAIEmbeddings(
  model="models/text-embedding-004",
  google_api_key=settings.GEMINI_API_KEY 
)

QDRANT_HOST_LOCAL = "localhost"
QDRANT_URL = f"http://{QDRANT_HOST_LOCAL}:6333"

COLLECTION_NAME = "ai_evaluation_context"
DOCS_DIR = "ground_truth_docs"

DOCUMENTS_TO_INGEST = [
  ("job_description.pdf", "job_description"),
  ("case_study.pdf", "case_study"),
  ("scoring_rubric.pdf", "scoring_rubric"),
]

def load_and_split_docs(file_path: str, doc_type: str):
  """Load and split documents into chunks."""
  print(f"loading and splitting {doc_type} from {file_path}")
  try:
    loader = PyPDFLoader(file_path)
    documents = loader.load()
  except Exception as e:
    print(f"Error loading {file_path}: {e}")
    return []
  
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
  )

  chunks = text_splitter.split_documents(documents)

  for chunk in chunks:
    chunk.metadata = {"type": doc_type}

  print(f"Splitting done. Total chunks: {len(chunks)}")
  return chunks

def ingest_documents():
  """Ingestion document into Qdrant vector store."""
  all_chunks = []

  os.makedirs(DOCS_DIR, exist_ok=True)
  print(f"Ensured directory exists: {DOCS_DIR}")

  for file_name, doc_type in DOCUMENTS_TO_INGEST:
    file_path = os.path.join(DOCS_DIR, file_name)
    if os.path.exists(file_path):
      all_chunks.extend(load_and_split_docs(file_path, doc_type))
    else:
      print(f"Warning: {file_path} does not exist. Skipping.")
    
  if not all_chunks:
    print("No documents to ingest. Exiting.")
    return
  
  print(f"\nTotal chunks for ingestion: {len(all_chunks)}")
  print(f"Starting Qdrant ingestion to {COLLECTION_NAME}...")

  Qdrant.from_documents(
    all_chunks,
    embeddings,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
    force_recreate=True
  )

  print("Ingestion completed.")

if __name__ == "__main__":
  ingest_documents()
