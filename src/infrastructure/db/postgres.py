import time

import psycopg2
from loguru import logger

from src.config.settings import settings


def connect_to_postgres(max_retries: int = 3, retry_delay: int = 3, connect_timeout: int = 5):
    """Connect to PostgreSQL with retry and return a live connection."""
    for attempt in range(1, max_retries + 1):
        conn = None
        try:
            logger.info(f"Postgres connection attempt {attempt}/{max_retries}")
            conn = psycopg2.connect(
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                dbname=settings.POSTGRES_DB,
                connect_timeout=connect_timeout,
            )
            conn.autocommit = True
            return conn
        except psycopg2.OperationalError as exc:
            logger.warning(f"Postgres connect failed: {exc}")
            if conn:
                conn.close()
            if attempt == max_retries:
                raise
            time.sleep(retry_delay)
