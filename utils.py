import logging
import litellm
import math
from config import config

logger = logging.getLogger("Utils")

def get_embedding(text: str) -> list:
    """Generate high-dimensional vector embeddings utilizing LiteLLM"""
    try:
        response = litellm.embedding(model=config.EMBEDDING_MODEL, input=[text])
        return response.data[0]["embedding"]
    except Exception as e:
        logger.warning(f"Failed to generate embedding: {e}. Generating fallback mock vector.")
        import random
        random.seed(hash(text))
        return [random.uniform(-1, 1) for _ in range(config.VECTOR_DIMENSION)]

def cosine_similarity(v1: list, v2: list) -> float:
    """Calculate the cosine similarity between two dimensional vectors"""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum(a * a for a in v1))
    magnitude_v2 = math.sqrt(sum(b * b for b in v2))
    if not magnitude_v1 or not magnitude_v2:
        return 0.0
    return dot_product / (magnitude_v1 * magnitude_v2)
