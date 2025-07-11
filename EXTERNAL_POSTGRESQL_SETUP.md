# External PostgreSQL Setup Guide

This guide helps you configure DocuMind AI Assistant to use an external PostgreSQL database.

## Configuration Steps

### 1. Update Environment Variables

Edit your `.env` file with your external PostgreSQL connection details:

```bash
# Database
DATABASE_URL="postgresql://username:password@host:port/database"
```

Replace the placeholders with your actual values:

- `username`: Your PostgreSQL username
- `password`: Your PostgreSQL password
- `host`: Your PostgreSQL server host/IP
- `port`: Your PostgreSQL port (usually 5432)
- `database`: Your database name

### 2. Example Configurations

#### Local PostgreSQL Installation

```bash
DATABASE_URL="postgresql://postgres:mypassword@localhost:5432/documind"
```

#### Remote PostgreSQL Server

```bash
DATABASE_URL="postgresql://myuser:mypassword@192.168.1.100:5432/documind"
```

#### Cloud PostgreSQL (e.g., AWS RDS)

```bash
DATABASE_URL="postgresql://username:password@my-db-instance.region.rds.amazonaws.com:5432/documind"
```

### 3. Test Connection

Test your database connection:

```bash
python -c "
from app.database import engine
from sqlalchemy import text
try:
    with engine.connect() as conn:
        result = conn.execute(text('SELECT version()'))
        print('✅ Connected to PostgreSQL:', result.fetchone()[0])
except Exception as e:
    print('❌ Connection failed:', e)
"
```

### 4. Run Database Migrations

Create the database tables:

```bash
alembic upgrade head
```

### 5. Start the Application

```bash
python run.py
```

## Troubleshooting

### Connection Issues

- Verify PostgreSQL server is running
- Check firewall settings
- Ensure credentials are correct
- Verify database exists

### Permission Issues

- Ensure user has CREATE, SELECT, INSERT, UPDATE, DELETE permissions
- Check if database exists and user has access

### SSL Issues

If your PostgreSQL requires SSL, add `?sslmode=require` to your connection string:

```bash
DATABASE_URL="postgresql://username:password@host:port/database?sslmode=require"
```

## Database Requirements

Your PostgreSQL database should support:

- PostgreSQL 12 or higher
- UTF-8 encoding
- User with sufficient permissions to create tables and indexes

## Migration from SQLite

If you have existing data in SQLite, you can migrate it using the migration script:

```bash
python migrate_to_postgres.py
```

This will transfer all your existing data to the external PostgreSQL database.
