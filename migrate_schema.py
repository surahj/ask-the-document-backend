#!/usr/bin/env python3
"""
Custom migration script for schema changes
Use this for complex migrations that can't be done with simple drop/recreate
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.database import engine, SessionLocal
from sqlalchemy import text


def migrate_schema():
    """Run custom schema migrations"""
    db = SessionLocal()

    try:
        # Example: Add a new column
        # db.execute(text("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS new_column VARCHAR(100)"))

        # Example: Create new index
        # db.execute(text("CREATE INDEX IF NOT EXISTS idx_document_chunks_content ON document_chunks(content)"))

        # Example: Update existing data
        # db.execute(text("UPDATE document_chunks SET new_column = 'default_value' WHERE new_column IS NULL"))

        db.commit()
        print("Schema migration completed successfully")

    except Exception as e:
        db.rollback()
        print(f"Migration failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    migrate_schema()
