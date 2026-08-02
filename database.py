import logging
from contextlib import contextmanager
from psycopg_pool import ConnectionPool
from config import config

logger = logging.getLogger("DatabaseManager")

class DatabaseManager:
    def __init__(self):
        self.pool = None

    def init_pool(self):
        """Initialize the connection pool during app startup"""
        if self.pool is None:
            logger.info("Initializing PostgreSQL connection pool...")
            try:
                self.pool = ConnectionPool(
                    conninfo=config.DB_DSN,
                    min_size=2,
                    max_size=10,
                    open=True,
                    kwargs={"connect_timeout": 5}
                )
                logger.info("PostgreSQL connection pool initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to create database connection pool: {e}")
                raise e

    def close_pool(self):
        """Close the connection pool during app shutdown"""
        if self.pool:
            logger.info("Closing PostgreSQL connection pool...")
            self.pool.close()
            self.pool = None
            logger.info("PostgreSQL connection pool closed.")

    @contextmanager
    def get_connection(self):
        """Retrieve a connection from the pool inside a thread-safe context"""
        if self.pool is None:
            self.init_pool()
        with self.pool.connection() as conn:
            yield conn

    def init_db(self):
        """Verify schemas and establish performance indexes"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
                    
                    # Create past reports table
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS past_reports (
                            id SERIAL PRIMARY KEY,
                            topic TEXT UNIQUE,
                            report TEXT,
                            embedding vector({config.VECTOR_DIMENSION})
                        );
                    """)
                    
                    # Create hypothesis evaluations table
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS hypothesis_evaluations (
                            id SERIAL PRIMARY KEY,
                            hypothesis TEXT UNIQUE,
                            domain TEXT,
                            core_claim TEXT,
                            underlying_assumptions TEXT[],
                            causal_chain TEXT[],
                            supporting_evidence TEXT,
                            counter_evidence TEXT,
                            vulnerability_score INT,
                            evaluation_summary TEXT,
                            critical_weaknesses TEXT[],
                            proposed_validation_protocol TEXT,
                            embedding vector({config.VECTOR_DIMENSION})
                        );
                    """)
                    
                    # Alter table to add the new multi-dimensional scores (safety migrations)
                    cur.execute("""
                        ALTER TABLE hypothesis_evaluations 
                        ADD COLUMN IF NOT EXISTS empirical_evidence_score INT,
                        ADD COLUMN IF NOT EXISTS logical_consistency_score INT,
                        ADD COLUMN IF NOT EXISTS confounder_vulnerability_score INT,
                        ADD COLUMN IF NOT EXISTS methodological_feasibility_score INT,
                        ADD COLUMN IF NOT EXISTS conversation_history JSONB,
                        ADD COLUMN IF NOT EXISTS expected_effect_size TEXT,
                        ADD COLUMN IF NOT EXISTS statistical_power_estimation TEXT,
                        ADD COLUMN IF NOT EXISTS scientific_consensus_index FLOAT,
                        ADD COLUMN IF NOT EXISTS bias_vulnerability_score INT;
                    """)
                    # Supabase Auth owns user records; these columns only store its UUID.
                    # The migration preserves any existing local development data.
                    cur.execute("""
                        ALTER TABLE hypothesis_evaluations
                        ADD COLUMN IF NOT EXISTS user_id UUID,
                        ADD COLUMN IF NOT EXISTS conversation_id UUID DEFAULT gen_random_uuid(),
                        ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
                        UPDATE hypothesis_evaluations
                        SET conversation_id = gen_random_uuid()
                        WHERE conversation_id IS NULL;
                        ALTER TABLE hypothesis_evaluations
                        DROP CONSTRAINT IF EXISTS hypothesis_evaluations_hypothesis_key;
                        CREATE UNIQUE INDEX IF NOT EXISTS hypothesis_evaluations_user_hypothesis_key
                        ON hypothesis_evaluations (user_id, hypothesis);
                        CREATE UNIQUE INDEX IF NOT EXISTS hypothesis_evaluations_conversation_id_key
                        ON hypothesis_evaluations (conversation_id);
                        CREATE INDEX IF NOT EXISTS hypothesis_evaluations_user_created_idx
                        ON hypothesis_evaluations (user_id, created_at DESC);
                    """)
                    
                    # Note: We omit index creation (HNSW) here because pgvector has a limit
                    # of 2000 dimensions for HNSW indexes, while our embeddings are 3072.
                    # Sequential scans are extremely fast for datasets under 100k rows.
                    pass
                conn.commit()
            logger.info("PostgreSQL database tables and vector indexes initialized successfully.")
        except Exception as e:
            logger.error(f"Postgres schema migration failed: {e}")
            raise e

# Singleton connection pool instance
db_manager = DatabaseManager()
