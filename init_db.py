import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from config import (
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_database():
    """Initialize the PostgreSQL database."""
    try:
        # Connect to PostgreSQL server
        conn = psycopg2.connect(
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (POSTGRES_DB,))
        exists = cursor.fetchone()
        
        if not exists:
            # Create database
            logger.info(f"Creating database {POSTGRES_DB}...")
            cursor.execute(f'CREATE DATABASE {POSTGRES_DB}')
            logger.info("Database created successfully!")
        else:
            logger.info(f"Database {POSTGRES_DB} already exists.")
            
        cursor.close()
        conn.close()
        
        # Initialize tables
        from database.db_manager import DatabaseManager
        db_manager = DatabaseManager()
        db_manager.init_db()
        logger.info("Database tables initialized successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return False

if __name__ == "__main__":
    init_database() 