import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    
    # Architecture:
    # - Cohere Command-R Plus: Research, paper citations, theory breakdown, advocate, adversary, arbiter
    # - Groq Llama-3.3 70B: Fast AI Copilot chat & tasks
    # - Gemini 2.5 Flash: Reserved strictly as secondary/tertiary fallback
    # - Gemini Embedding 001: 3072-dimensional vector embeddings
    THEORY_MODEL: str = os.getenv("THEORY_MODEL", "cohere/command-r-plus-08-2024")
    ADVOCATE_MODEL: str = os.getenv("ADVOCATE_MODEL", "cohere/command-r-plus-08-2024")
    ADVERSARY_MODEL: str = os.getenv("ADVERSARY_MODEL", "cohere/command-r-plus-08-2024")
    ARBITER_MODEL: str = os.getenv("ARBITER_MODEL", "cohere/command-r-plus-08-2024")
    
    PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", "cohere/command-r-plus-08-2024")
    JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "cohere/command-r-plus-08-2024")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini/gemini-embedding-001")
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "groq/llama-3.3-70b-versatile")
    
    # Data Dimensions
    VECTOR_DIMENSION: int = 3072

    # Database Configuration (PostgreSQL)
    DB_DSN: str = os.getenv(
        "DB_DSN", 
        "dbname=research_db user=postgres password=postgres host=localhost port=5433"
    )

    # Cache Configuration (Redis)
    REDIS_URL: str = os.getenv("REDIS_URL", "")
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))

# Singleton instance
config = Config()
