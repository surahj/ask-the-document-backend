"""
Database connection and session management with PostgreSQL pgvector support
"""

from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

# Create database engine with PostgreSQL support
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.debug,  # Enable SQL logging in debug mode
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all database tables and enable pgvector extension"""
    try:
        # Create tables
        Base.metadata.create_all(bind=engine)

        # Enable pgvector extension if using PostgreSQL
        if "postgresql" in settings.database_url:
            with engine.connect() as conn:
                # Enable pgvector extension
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                print("pgvector extension enabled successfully")

        print("Database tables created successfully")

    except Exception as e:
        print(f"Error creating tables: {e}")
        raise


def drop_tables():
    """Drop all database tables"""
    try:
        Base.metadata.drop_all(bind=engine)
        print("Database tables dropped successfully")
    except Exception as e:
        print(f"Error dropping tables: {e}")
        raise


def check_pgvector_extension():
    """Check if pgvector extension is available"""
    try:
        if "postgresql" in settings.database_url:
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT * FROM pg_extension WHERE extname = 'vector'")
                )
                return result.fetchone() is not None
        return False
    except Exception as e:
        print(f"Error checking pgvector extension: {e}")
        return False


def create_vector_indexes():
    """Create vector indexes for better search performance"""
    try:
        if "postgresql" in settings.database_url and check_pgvector_extension():
            with engine.connect() as conn:
                # Create index for document chunks embeddings
                conn.execute(
                    text(
                        """
                    CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding 
                    ON document_chunks 
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """
                    )
                )
                conn.commit()
                print("Vector indexes created successfully")
        else:
            print("pgvector extension not available, skipping vector indexes")
    except Exception as e:
        print(f"Error creating vector indexes: {e}")


def initialize_database():
    """Initialize database with all required components"""
    try:
        create_tables()
        create_vector_indexes()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise
