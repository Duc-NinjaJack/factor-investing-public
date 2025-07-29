"""
Common Database Connection Module

This module provides a unified database connection interface for all codes in the project.
It supports both SQLAlchemy and PyMySQL connections with proper configuration management.

Usage:
    from production.database import get_engine, get_connection, DatabaseManager
    
    # Get SQLAlchemy engine
    engine = get_engine()
    
    # Get PyMySQL connection
    conn = get_connection()
    
    # Use DatabaseManager for advanced features
    db_manager = DatabaseManager()
    engine = db_manager.get_engine()
    conn = db_manager.get_connection()
"""

from .connection import DatabaseManager, get_engine, get_connection

__all__ = ['DatabaseManager', 'get_engine', 'get_connection']