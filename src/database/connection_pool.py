#!/usr/bin/env python3
"""
Database Connection Pool Manager for Beverly Knits ERP
Provides efficient connection pooling with psycopg2
"""

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.secure_config import SecureConfig

logger = logging.getLogger(__name__)


class DatabasePool:
    """Singleton database connection pool manager"""
    
    _instance = None
    _pool = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize connection pool if not already done"""
        if DatabasePool._pool is None:
            self.initialize_pool()
    
    def initialize_pool(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the connection pool"""
        if config is None:
            config = SecureConfig.get_database_config()
        
        try:
            DatabasePool._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=config.get('connection_pool_size', 10),
                host=config['localhost'],
                port=config['5432'],
                database=config['beverly_knits_erp'],
                user=config['erp_user'],
                password=config['erp_password']
            )
            logger.info(f"Database connection pool initialized with {config.get('connection_pool_size', 10)} connections")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self, dict_cursor: bool = True):
        """
        Get a connection from the pool with context manager support
        
        Args:
            dict_cursor: If True, use RealDictCursor for JSON-friendly results
            
        Yields:
            Database connection
        """
        conn = None
        try:
            conn = DatabasePool._pool.getconn()
            if dict_cursor:
                conn.cursor_factory = RealDictCursor
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                DatabasePool._pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self, dict_cursor: bool = True):
        """
        Get a cursor from a pooled connection with context manager support
        
        Args:
            dict_cursor: If True, use RealDictCursor for JSON-friendly results
            
        Yields:
            Database cursor
        """
        with self.get_connection(dict_cursor) as conn:
            cursor = conn.cursor()
            try:
                yield cursor
            finally:
                cursor.close()
    
    def execute_query(self, query: str, params: tuple = None, fetch_one: bool = False):
        """
        Execute a query and return results
        
        Args:
            query: SQL query to execute
            params: Query parameters
            fetch_one: If True, return only one result
            
        Returns:
            Query results
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            if fetch_one:
                return cursor.fetchone()
            return cursor.fetchall()
    
    def execute_many(self, query: str, params_list: list):
        """
        Execute a query with multiple parameter sets
        
        Args:
            query: SQL query to execute
            params_list: List of parameter tuples
        """
        with self.get_cursor() as cursor:
            cursor.executemany(query, params_list)
    
    def close_pool(self):
        """Close all connections in the pool"""
        if DatabasePool._pool:
            DatabasePool._pool.closeall()
            logger.info("Database connection pool closed")
            DatabasePool._pool = None
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get current pool status"""
        if DatabasePool._pool:
            return {
                'active': True,
                'min_connections': DatabasePool._pool.minconn,
                'max_connections': DatabasePool._pool.maxconn
            }
        return {'active': False}


# Global pool instance
db_pool = DatabasePool()


# Convenience functions
def get_connection(dict_cursor: bool = True):
    """Get a database connection from the pool"""
    return db_pool.get_connection(dict_cursor)


def get_cursor(dict_cursor: bool = True):
    """Get a database cursor from the pool"""
    return db_pool.get_cursor(dict_cursor)


def execute_query(query: str, params: tuple = None, fetch_one: bool = False):
    """Execute a query using the pool"""
    return db_pool.execute_query(query, params, fetch_one)


def close_pool():
    """Close the connection pool"""
    db_pool.close_pool()


if __name__ == "__main__":
    # Test the connection pool
    print("Testing Database Connection Pool...")
    
    try:
        # Initialize pool
        pool = DatabasePool()
        print(f"Pool status: {pool.get_pool_status()}")
        
        # Test query
        with pool.get_cursor() as cursor:
            cursor.execute("SELECT 1 as test")
            result = cursor.fetchone()
            print(f"Test query result: {result}")
        
        # Test multiple connections
        print("\nTesting multiple connections...")
        for i in range(3):
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT %s as conn_test", (i,))
                result = cursor.fetchone()
                print(f"Connection {i}: {result}")
                cursor.close()
        
        print("\nConnection pool test successful!")
        
    except Exception as e:
        print(f"Connection pool test failed: {e}")
    finally:
        pool.close_pool()