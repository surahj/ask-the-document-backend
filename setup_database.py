#!/usr/bin/env python3
"""
Database setup script for DocuMind AI Assistant with PostgreSQL pgvector
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from app.config import settings
from app.database import initialize_database, check_pgvector_extension


def setup_postgresql():
    """Setup PostgreSQL database with pgvector extension"""
    try:
        print("Setting up PostgreSQL database with pgvector...")

        # Test database connection
        engine = create_engine(settings.database_url)

        with engine.connect() as conn:
            # Check if pgvector extension is available
            result = conn.execute(
                text("SELECT * FROM pg_available_extensions WHERE name = 'vector'")
            )
            if not result.fetchone():
                print("ERROR: pgvector extension is not available in PostgreSQL")
                print("Please install pgvector extension:")
                print(
                    "  - For Ubuntu/Debian: sudo apt-get install postgresql-14-pgvector"
                )
                print("  - For macOS: brew install pgvector")
                print("  - For Docker: Use postgres:15-pgvector image")
                return False

            # Enable pgvector extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            print("✓ pgvector extension enabled")

            # Create database tables
            initialize_database()
            print("✓ Database tables created")

            # Create vector indexes
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
            print("✓ Vector indexes created")

            print("✓ Database setup completed successfully!")
            return True

    except OperationalError as e:
        print(f"ERROR: Database connection failed: {e}")
        print("Please check your database configuration in .env file")
        return False
    except Exception as e:
        print(f"ERROR: Database setup failed: {e}")
        return False


def setup_sqlite():
    """Setup SQLite database (fallback)"""
    try:
        print("Setting up SQLite database...")

        # Update settings for SQLite
        settings.use_postgresql = False
        settings.database_url = "sqlite:///./documind.db"

        # Initialize database
        initialize_database()
        print("✓ SQLite database setup completed!")
        return True

    except Exception as e:
        print(f"ERROR: SQLite setup failed: {e}")
        return False


def main():
    """Main setup function"""
    print("DocuMind AI Assistant - Database Setup")
    print("=" * 50)

    # Use PostgreSQL with pgvector
    print("Setting up PostgreSQL with pgvector...")
    success = setup_postgresql()

    if success:
        print("\n✓ Database setup completed successfully!")
        print("\nNext steps:")
        print("1. Start the application: python -m uvicorn app.main:app --reload")
        print("2. Access the API documentation at: http://localhost:8000/docs")
        print("3. Upload documents and start asking questions!")
    else:
        print("\n✗ Database setup failed!")
        print("Please check the error messages above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
