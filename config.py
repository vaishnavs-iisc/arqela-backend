import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    
    # Model selections (Gemini 2.5 Flash is ultra-fast, robust, and handles structured output natively)
    PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", "gemini/gemini-2.5-flash")
    JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "gemini/gemini-2.5-flash")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini/gemini-embedding-001")
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gemini/gemini-2.5-flash")
    
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
