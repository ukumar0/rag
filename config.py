# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# PostgreSQL Configuration
POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'rag_test_db')

# Database URL
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Model Configuration
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model
LLM_MODEL = "gpt-3.5-turbo"  # OpenAI chat model

# Chunking Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval Configuration
TOP_K_CHUNKS = 5
SIMILARITY_THRESHOLD = 0.7  # Lowered from 0.7 to be less restrictive

# Weaviate Configuration
WEAVIATE_URL = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')  # For WCS or authenticated instances
WEAVIATE_INDEX_NAME = os.getenv('WEAVIATE_INDEX_NAME', 'RagDocuments')
