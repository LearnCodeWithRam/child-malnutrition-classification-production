"""
Production Database Engine and Session Management.
Supports PostgreSQL with connection pooling and a test SQLite fallback.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool

DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://poshan_user:poshan_secure_password_2026@postgres:5432/poshan_db"
)

Base = declarative_base()

import socket
from urllib.parse import urlparse

def get_engine(db_url: str = None):
    url = db_url or os.getenv(
        "DATABASE_URL", 
        "postgresql://poshan_user:poshan_secure_password_2026@postgres:5432/poshan_db"
    )
    if url.startswith("postgresql"):
        parsed = urlparse(url)
        hostname = parsed.hostname or "localhost"
        
        # Fast DNS check to avoid 10s Windows NetBIOS lookup timeout if host does not exist locally
        try:
            socket.gethostbyname(hostname)
            host_reachable = True
        except Exception:
            host_reachable = False

        if host_reachable:
            try:
                eng = create_engine(
                    url,
                    poolclass=QueuePool,
                    pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
                    max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
                    pool_pre_ping=True,
                    pool_recycle=1800,
                    connect_args={"connect_timeout": 1}
                )
                with eng.connect() as conn:
                    pass
                return eng
            except Exception as e:
                pass

        # Graceful SQLite fallback for local developer testing
        fallback_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "poshan_local.db")
        print(f"ℹ️ [Database] PostgreSQL host '{hostname}' not reachable. Using local SQLite: {fallback_path}")
        return create_engine(f"sqlite:///{fallback_path}", connect_args={"check_same_thread": False})
    
    return create_engine(url, connect_args={"check_same_thread": False})

_engine = None

def get_db_engine(db_url: str = None):
    global _engine
    if _engine is None or db_url is not None:
        _engine = get_engine(db_url)
    return _engine

def get_session_factory(engine=None):
    eng = engine or get_db_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=eng)

def SessionLocal():
    return get_session_factory()()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db(engine=None):
    """Create all registered tables."""
    eng = engine or get_db_engine()
    Base.metadata.create_all(bind=eng)
    return eng
