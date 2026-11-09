import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text, URL
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")
load_dotenv(Path.cwd() / ".env")

# Database configuration
RAW_DB_URL = os.getenv("DATABASE_URL")

if RAW_DB_URL:
    if RAW_DB_URL.startswith("postgresql://") and "+psycopg" not in RAW_DB_URL:
        DATABASE_URL = RAW_DB_URL.replace("postgresql://", "postgresql+psycopg://", 1)
    elif RAW_DB_URL.startswith("postgres://"):
        DATABASE_URL = RAW_DB_URL.replace("postgres://", "postgresql+psycopg://", 1)
    else:
        DATABASE_URL = RAW_DB_URL
else:
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    if not DB_PASSWORD:
        raise ValueError("DATABASE_URL or DB_PASSWORD missing in environment/.env")

    DB_USER = os.getenv("DB_USER", "postgres")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", "5432"))
    DB_NAME = os.getenv("DB_NAME", "sentinelrag")

    DATABASE_URL = URL.create(
        drivername="postgresql+psycopg",
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
    )

# Engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Session
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)

# Base
class Base(DeclarativeBase):
    pass

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Test connection
def test_connection():
    with engine.connect() as connection:
        result = connection.execute(text("SELECT version()"))
        print("Database connected successfully!")
        print(result.scalar())

# Initialize pgvector
def init_database():
    with engine.begin() as connection:
        connection.execute(
            text("CREATE EXTENSION IF NOT EXISTS vector")
        )
    print("pgvector initialized successfully!")

if __name__ == "__main__":
    test_connection()