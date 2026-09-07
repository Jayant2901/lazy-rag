import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GEN_MODEL = os.environ.get("LAZY_RAG_MODEL", "claude-haiku-4-5-20251001")
EMBED_MODEL = os.environ.get("LAZY_RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
