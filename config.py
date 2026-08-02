import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    
    # Pure Cohere + Groq High-Speed Architecture:
    # - Cohere Command-R (08-2024): High-speed scientific research & citations (~1.2s response time)
    # - Cohere Embed English v3.0: 1024-dimensional scientific embeddings
    # - Groq Llama-3.3 70B: Fast AI Copilot chat & tasks
    THEORY_MODEL: str = os.getenv("THEORY_MODEL", "cohere/command-r-08-2024")
    ADVOCATE_MODEL: str = os.getenv("ADVOCATE_MODEL", "cohere/command-r-08-2024")
    ADVERSARY_MODEL: str = os.getenv("ADVERSARY_MODEL", "cohere/command-r-08-2024")
    ARBITER_MODEL: str = os.getenv("ARBITER_MODEL", "cohere/command-r-08-2024")
    
    PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", "cohere/command-r-08-2024")
    JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "cohere/command-r-08-2024")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "cohere/embed-english-v3.0")
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "groq/llama-3.3-70b-versatile")
    
    # Data Dimensions
    VECTOR_DIMENSION: int = 1024

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
