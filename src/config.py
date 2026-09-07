import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEN_MODEL = os.environ.get("LAZY_RAG_MODEL", "openai/gpt-oss-120b")
EMBED_MODEL = os.environ.get("LAZY_RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
