"""
Database Connection Management

This module provides a unified interface for database connections across the project.
It supports both SQLAlchemy engines and PyMySQL connections with proper configuration
management, connection pooling, and error handling.

Author: Factor Investing Team
Date: 2025-07-30
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
from contextlib import contextmanager
import warnings

# Database drivers
try:
    import pymysql
    from pymysql.cursors import DictCursor
    PYMYSQL_AVAILABLE = True
except ImportError:
    PYMYSQL_AVAILABLE = False
    warnings.warn("PyMySQL not available. PyMySQL connections will not work.")

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import Engine
    from sqlalchemy.pool import QueuePool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    warnings.warn("SQLAlchemy not available. SQLAlchemy engines will not work.")

# Global connection cache
_engine_cache = {}
_connection_cache = {}

class DatabaseConfigError(Exception):
    """Raised when there's an issue with database configuration."""
    pass

class DatabaseConnectionError(Exception):
    """Raised when there's an issue establishing database connections."""
    pass

class DatabaseManager:
    """
    Centralized database connection manager.
    
    This class provides a unified interface for managing database connections
    across the entire project. It supports both SQLAlchemy engines and PyMySQL
    connections with proper configuration management and connection pooling.
    
    Features:
    - Automatic configuration loading from database.yml
    - Connection pooling and caching
    - Environment-specific configurations (production/development)
    - Error handling and logging
    - Context manager support
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 environment: str = 'production',
                 enable_pooling: bool = True,
                 pool_size: int = 10,
                 max_overflow: int = 20,
                 pool_timeout: int = 30,
                 pool_recycle: int = 3600):
        """
        Initialize the database manager.
        
        Args:
            config_path: Path to database configuration file
            environment: Environment to use ('production' or 'development')
            enable_pooling: Whether to enable connection pooling
            pool_size: Number of connections to maintain in pool
            max_overflow: Maximum number of connections beyond pool_size
            pool_timeout: Timeout for getting connection from pool
            pool_recycle: Recycle connections after this many seconds
        """
        self.environment = environment
        self.enable_pooling = enable_pooling
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize connection caches
        self._engine = None
        self._connection = None
        
        self.logger.info(f"DatabaseManager initialized for environment: {environment}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the database manager."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load database configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Database configuration dictionary
            
        Raises:
            DatabaseConfigError: If configuration cannot be loaded
        """
        try:
            # Determine config path
            if config_path is None:
                # Try to find config relative to project root
                project_root = self._find_project_root()
                config_path = project_root / 'config' / 'database.yml'
            
            config_path = Path(config_path)
            
            if not config_path.exists():
                raise DatabaseConfigError(f"Database configuration file not found: {config_path}")
            
            # Load configuration
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if self.environment not in config_data:
                raise DatabaseConfigError(f"Environment '{self.environment}' not found in configuration")
            
            config = config_data[self.environment]
            
            # Validate required fields
            required_fields = ['host', 'schema_name', 'username', 'password']
            for field in required_fields:
                if field not in config:
                    raise DatabaseConfigError(f"Required field '{field}' missing from configuration")
            
            self.logger.info(f"Database configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load database configuration: {e}")
            raise DatabaseConfigError(f"Configuration loading failed: {e}")
    
    def _find_project_root(self) -> Path:
        """Find the project root directory."""
        current_path = Path.cwd()
        
        # Look for config directory in current path and parents
        while current_path != current_path.parent:
            if (current_path / 'config' / 'database.yml').exists():
                return current_path
            current_path = current_path.parent
        
        # If not found, assume current directory is project root
        return Path.cwd()
    
    def get_engine(self, force_new: bool = False) -> Engine:
        """
        Get SQLAlchemy engine with connection pooling.
        
        Args:
            force_new: Force creation of new engine (bypass cache)
            
        Returns:
            SQLAlchemy engine
            
        Raises:
            DatabaseConnectionError: If engine creation fails
        """
        if not SQLALCHEMY_AVAILABLE:
            raise DatabaseConnectionError("SQLAlchemy not available")
        
        # Check cache
        cache_key = f"engine_{self.environment}"
        if not force_new and cache_key in _engine_cache:
            self.logger.debug("Returning cached SQLAlchemy engine")
            return _engine_cache[cache_key]
        
        try:
            # Build connection string
            connection_string = (
                f"mysql+pymysql://{self.config['username']}:{self.config['password']}@"
                f"{self.config['host']}/{self.config['schema_name']}"
            )
            
            # Engine parameters
            engine_params = {
                'echo': False,  # Set to True for SQL debugging
                'pool_pre_ping': True,  # Validate connections before use
            }
            
            # Add pooling parameters if enabled
            if self.enable_pooling:
                engine_params.update({
                    'poolclass': QueuePool,
                    'pool_size': self.pool_size,
                    'max_overflow': self.max_overflow,
                    'pool_timeout': self.pool_timeout,
                    'pool_recycle': self.pool_recycle,
                })
            
            # Create engine
            engine = create_engine(connection_string, **engine_params)
            
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            # Cache engine
            _engine_cache[cache_key] = engine
            self._engine = engine
            
            self.logger.info("SQLAlchemy engine created successfully")
            return engine
            
        except Exception as e:
            self.logger.error(f"Failed to create SQLAlchemy engine: {e}")
            raise DatabaseConnectionError(f"Engine creation failed: {e}")
    
    def get_connection(self, force_new: bool = False) -> pymysql.Connection:
        """
        Get PyMySQL connection.
        
        Args:
            force_new: Force creation of new connection (bypass cache)
            
        Returns:
            PyMySQL connection
            
        Raises:
            DatabaseConnectionError: If connection creation fails
        """
        if not PYMYSQL_AVAILABLE:
            raise DatabaseConnectionError("PyMySQL not available")
        
        # Check cache
        cache_key = f"connection_{self.environment}"
        if not force_new and cache_key in _connection_cache:
            cached_conn = _connection_cache[cache_key]
            try:
                # Test if cached connection is still alive
                cached_conn.ping(reconnect=False)
                self.logger.debug("Returning cached PyMySQL connection")
                return cached_conn
            except:
                # Remove dead connection from cache
                del _connection_cache[cache_key]
        
        try:
            # Create connection
            connection = pymysql.connect(
                host=self.config['host'],
                user=self.config['username'],
                password=self.config['password'],
                database=self.config['schema_name'],
                charset='utf8mb4',
                cursorclass=DictCursor,
                autocommit=False,
                connect_timeout=30,
                read_timeout=30,
                write_timeout=30
            )
            
            # Cache connection
            _connection_cache[cache_key] = connection
            self._connection = connection
            
            self.logger.info("PyMySQL connection created successfully")
            return connection
            
        except Exception as e:
            self.logger.error(f"Failed to create PyMySQL connection: {e}")
            raise DatabaseConnectionError(f"Connection creation failed: {e}")
    
    @contextmanager
    def get_engine_context(self):
        """Context manager for SQLAlchemy engine."""
        engine = self.get_engine()
        try:
            yield engine
        except Exception as e:
            self.logger.error(f"Error in engine context: {e}")
            raise
        finally:
            # Engine cleanup is handled by SQLAlchemy
            pass
    
    @contextmanager
    def get_connection_context(self):
        """Context manager for PyMySQL connection."""
        connection = self.get_connection()
        try:
            yield connection
        except Exception as e:
            self.logger.error(f"Error in connection context: {e}")
            connection.rollback()
            raise
        finally:
            # Don't close connection here as it's cached
            pass
    
    def test_connection(self) -> bool:
        """
        Test database connectivity.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test SQLAlchemy engine
            engine = self.get_engine()
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            # Test PyMySQL connection
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            
            self.logger.info("Database connectivity test successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Database connectivity test failed: {e}")
            return False
    
    def close_all_connections(self):
        """Close all cached connections."""
        try:
            # Close PyMySQL connections
            for cache_key, connection in list(_connection_cache.items()):
                try:
                    connection.close()
                    self.logger.debug(f"Closed connection: {cache_key}")
                except:
                    pass
                del _connection_cache[cache_key]
            
            # SQLAlchemy engines handle their own cleanup
            _engine_cache.clear()
            
            self.logger.info("All database connections closed")
            
        except Exception as e:
            self.logger.error(f"Error closing connections: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current database configuration."""
        return self.config.copy()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close_all_connections()
        except:
            pass

# Global database manager instance
_global_db_manager = None

def get_database_manager(environment: str = 'production', **kwargs) -> DatabaseManager:
    """
    Get global database manager instance.
    
    Args:
        environment: Environment to use
        **kwargs: Additional arguments for DatabaseManager
        
    Returns:
        DatabaseManager instance
    """
    global _global_db_manager
    
    if _global_db_manager is None:
        _global_db_manager = DatabaseManager(environment=environment, **kwargs)
    
    return _global_db_manager

def get_engine(environment: str = 'production', **kwargs) -> Engine:
    """
    Get SQLAlchemy engine.
    
    Args:
        environment: Environment to use
        **kwargs: Additional arguments for DatabaseManager
        
    Returns:
        SQLAlchemy engine
    """
    db_manager = get_database_manager(environment=environment, **kwargs)
    return db_manager.get_engine()

def get_connection(environment: str = 'production', **kwargs) -> pymysql.Connection:
    """
    Get PyMySQL connection.
    
    Args:
        environment: Environment to use
        **kwargs: Additional arguments for DatabaseManager
        
    Returns:
        PyMySQL connection
    """
    db_manager = get_database_manager(environment=environment, **kwargs)
    return db_manager.get_connection()

def test_connection(environment: str = 'production', **kwargs) -> bool:
    """
    Test database connectivity.
    
    Args:
        environment: Environment to use
        **kwargs: Additional arguments for DatabaseManager
        
    Returns:
        True if connection successful, False otherwise
    """
    db_manager = get_database_manager(environment=environment, **kwargs)
    return db_manager.test_connection()

def close_all_connections():
    """Close all database connections."""
    global _global_db_manager
    if _global_db_manager:
        _global_db_manager.close_all_connections()
        _global_db_manager = None