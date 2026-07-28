import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TOP_K = 10
RELEVANCE_THRESHOLD = 0.7

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 50
RERANK_TOP_K = 5

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "equity_research"