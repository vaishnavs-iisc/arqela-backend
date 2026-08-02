import os
import json
import math
import time
import psycopg
import redis
from dotenv import load_dotenv
import litellm

# Load API keys
load_dotenv()

# ==========================================
# 1. Connection Configurations
# ==========================================
# Redis Client
r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# Postgres DSN
DB_DSN = "dbname=research_db user=postgres password=postgres host=localhost port=5433"

# Embedding model config
# Google's gemini-embedding-001 model generates vectors of dimension 3072
EMBEDDING_MODEL = "gemini/gemini-embedding-001"
VECTOR_DIMENSION = 3072

# Helper to get embeddings from LiteLLM
def get_embedding(text: str) -> list:
    try:
        response = litellm.embedding(
            model=EMBEDDING_MODEL,
            input=[text]
        )
        return response.data[0]["embedding"]
    except Exception as e:
        # If no API key is provided, we return a mock vector for testing
        print(f"[Warning] Failed to generate embedding: {e}. Generating mock vector...")
        # Simple pseudo-random mock vector based on string hash for local debugging
        import random
        random.seed(hash(text))
        return [random.uniform(-1, 1) for _ in range(VECTOR_DIMENSION)]

# ==========================================
# 2. Short-Term Memory (STM) Implementation
# ==========================================
class ShortTermMemory:
    """Manages chat session history in Redis."""
    def __init__(self, ttl: int = 3600):
        self.ttl = ttl

    def save_message(self, session_id: str, role: str, content: str):
        key = f"chat:{session_id}"
        message = json.dumps({"role": role, "content": content})
        r.rpush(key, message)
        r.expire(key, self.ttl)

    def get_chat_history(self, session_id: str) -> list:
        key = f"chat:{session_id}"
        raw_messages = r.lrange(key, 0, -1)
        return [json.loads(msg) for msg in raw_messages]

# ==========================================
# 3. Long-Term Memory (LTM) Implementation
# ==========================================
class LongTermMemory:
    """Manages permanent document storage with pgvector in PostgreSQL."""
    def __init__(self):
        self.init_db()

    def init_db(self):
        with psycopg.connect(DB_DSN) as conn:
            with conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                # Drop table if exists to update dimensions
                cur.execute("DROP TABLE IF EXISTS past_reports;")
                # Create table with a vector column
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS past_reports (
                        id SERIAL PRIMARY KEY,
                        topic TEXT UNIQUE,
                        report TEXT,
                        embedding vector({VECTOR_DIMENSION})
                    );
                """)
            conn.commit()

    def save_report(self, topic: str, report: str, embedding: list):
        with psycopg.connect(DB_DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO past_reports (topic, report, embedding)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (topic) DO UPDATE 
                    SET report = EXCLUDED.report, embedding = EXCLUDED.embedding;
                    """,
                    (topic, report, embedding)
                )
            conn.commit()

    def search_similar_reports(self, query_embedding: list, limit: int = 2) -> list:
        with psycopg.connect(DB_DSN) as conn:
            with conn.cursor() as cur:
                # pgvector cosine distance operator is '<=>'
                # Cosine distance = 1 - Cosine Similarity.
                # So order by embedding <=> query_embedding ASC (closes distance first)
                cur.execute(
                    f"""
                    SELECT topic, report, 1 - (embedding <=> %s::vector) as similarity
                    FROM past_reports
                    ORDER BY embedding <=> %s::vector ASC
                    LIMIT %s;
                    """,
                    (query_embedding, query_embedding, limit)
                )
                return cur.fetchall()

# ==========================================
# 4. Semantic Cache Implementation
# ==========================================
def cosine_similarity(v1: list, v2: list) -> float:
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum(a * a for a in v1))
    magnitude_v2 = math.sqrt(sum(b * b for b in v2))
    if not magnitude_v1 or not magnitude_v2:
        return 0.0
    return dot_product / (magnitude_v1 * magnitude_v2)

class SemanticCache:
    """Caches query results in Redis and checks cosine similarity of new queries."""
    def __init__(self, threshold: float = 0.90, ttl: int = 86400):
        self.threshold = threshold
        self.ttl = ttl

    def get(self, query: str) -> str:
        # 1. Get embedding for the new query
        query_vector = get_embedding(query)
        
        # 2. Scan all cached items in Redis
        # (In production, you'd use Redis Stack Vector Search, but this loop shows the math clearly!)
        keys = r.keys("cache:*")
        for key in keys:
            cached_data = json.loads(r.get(key))
            cached_vector = cached_data["embedding"]
            
            # 3. Calculate cosine similarity
            similarity = cosine_similarity(query_vector, cached_vector)
            print(f"[Cache Checked] Comparing '{query}' with '{cached_data['query']}' -> Similarity: {similarity:.2%}")
            
            if similarity >= self.threshold:
                print(f"[Cache HIT] Found similar query: '{cached_data['query']}' (Similarity: {similarity:.2%})")
                return cached_data["report"]
                
        print("[Cache MISS] No similar query found in cache.")
        return None

    def set(self, query: str, report: str):
        query_vector = get_embedding(query)
        key = f"cache:{hash(query)}"
        data = {
            "query": query,
            "report": report,
            "embedding": query_vector
        }
        r.set(key, json.dumps(data), ex=self.ttl)

# ==========================================
# 5. Demonstration Run
# ==========================================
if __name__ == "__main__":
    print("--- Phase 3 Memory Verification ---")
    
    # 1. Verify Short-Term Memory (Redis)
    print("\n1. Testing Short-Term Memory...")
    stm = ShortTermMemory()
    stm.save_message("session_abc", "user", "Hello agent!")
    stm.save_message("session_abc", "assistant", "Hello! How can I help you research today?")
    history = stm.get_chat_history("session_abc")
    print(f"Retrieved session history: {history}")

    # 2. Verify Long-Term Memory (Postgres + pgvector)
    print("\n2. Testing Long-Term Memory (pgvector)...")
    ltm = LongTermMemory()
    
    topic_1 = "Quantum Computing Basics"
    report_1 = "Quantum computers use qubits which can represent 0 and 1 simultaneously."
    v_1 = get_embedding(topic_1)
    
    topic_2 = "History of Space Flight"
    report_2 = "Sputnik 1 was the first artificial satellite launched into space in 1957."
    v_2 = get_embedding(topic_2)
    
    print("Saving past research reports to PostgreSQL...")
    ltm.save_report(topic_1, report_1, v_1)
    ltm.save_report(topic_2, report_2, v_2)
    
    # Let's search LTM with a semantic query
    search_query = "Tell me about quantum qubits"
    print(f"Searching LTM for: '{search_query}'")
    search_vector = get_embedding(search_query)
    results = ltm.search_similar_reports(search_vector, limit=1)
    for topic, report, similarity in results:
        print(f"Matched Topic: '{topic}' (Similarity: {similarity:.2%})")
        print(f"Content: {report}")

    # 3. Verify Semantic Cache
    print("\n3. Testing Semantic Cache...")
    cache = SemanticCache(threshold=0.85)
    
    query_a = "Explain quantum computing basics"
    fake_report = "This is a detailed generated report about Quantum Computing Basics..."
    
    # Set to cache
    print(f"Writing query '{query_a}' to cache...")
    cache.set(query_a, fake_report)
    
    # Query with a highly similar query (should Cache HIT)
    query_b = "Tell me about quantum computing basics"
    print(f"Querying cache with: '{query_b}'")
    cached_report = cache.get(query_b)
    print(f"Result: {cached_report}")
    
    # Query with a completely different query (should Cache MISS)
    query_c = "Who won the world cup in 2022?"
    print(f"Querying cache with: '{query_c}'")
    cached_report = cache.get(query_c)
