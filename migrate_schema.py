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
        # Add Cloudinary fields to documents table
        db.execute(
            text(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS cloudinary_url VARCHAR(1000)"
            )
        )
        db.execute(
            text(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS cloudinary_public_id VARCHAR(255)"
            )
        )

        # Make file_path nullable for Cloudinary compatibility
        db.execute(text("ALTER TABLE documents ALTER COLUMN file_path DROP NOT NULL"))

        db.commit()
        print("Schema migration completed successfully")
        print("Added Cloudinary fields to documents table")

    except Exception as e:
        db.rollback()
        print(f"Migration failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    migrate_schema()
