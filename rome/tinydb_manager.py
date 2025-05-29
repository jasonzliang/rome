"""
TinyDB management functionality separated from VersionManager.
Handles all database operations with improved efficiency and code reuse.
"""

import datetime
import os
import threading
from typing import List, Dict, Any, Optional, Union
from contextlib import contextmanager

from tinydb import TinyDB, Query
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware
import portalocker

from .logger import get_logger


class LockedJSONStorage(JSONStorage):
    """TinyDB storage with file locking support"""

    def __init__(self, path: str, create_dirs: bool = True, **kwargs):
        self.path = path
        if create_dirs:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        super().__init__(path, **kwargs)

    def read(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            portalocker.lock(f, portalocker.LOCK_SH)
            return super()._read(f)

    def write(self, data):
        with open(self.path, 'w', encoding='utf-8') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            super()._write(f, data)


class TinyDBManager:
    """
    Manages all TinyDB operations with optimized efficiency and code reuse.
    Provides a clean interface for database operations while handling
    connection management, query building, and data enrichment.
    """
    def __init__(self, get_db_path_func, config: Dict = None):
        self.config = config or {}
        self.logger = get_logger()
        self._get_db_path = get_db_path_func
        self._db_cache = {}  # Add connection cache
        self._cache_lock = threading.Lock()  # Thread safety

    def _get_database(self, file_path: str) -> TinyDB:
        """Get or create a cached database connection."""
        db_path = self._get_db_path(file_path)

        with self._cache_lock:
            if db_path not in self._db_cache:
                # FIXED: Properly instantiate storage with function call
                storage = CachingMiddleware(LockedJSONStorage)
                self._db_cache[db_path] = TinyDB(db_path, storage=storage)

            return self._db_cache[db_path]

    def _get_timestamp(self) -> str:
        """Generate ISO format timestamp."""
        return datetime.datetime.now().isoformat()

    @contextmanager
    def _db_context(self, file_path: str, table_name: str = None):
        """Context manager for database operations without closing cached connections."""
        db = self._get_database(file_path)
        try:
            if table_name:
                yield db.table(table_name)
            else:
                yield db
        except Exception:
            # Don't close cached connections on exceptions
            raise
        # FIXED: Removed db.close() to prevent closing cached connections

    def _enrich_data(self, data: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """Add standard metadata fields to data."""
        return {
            'timestamp': self._get_timestamp(),
            'file_path': file_path,
            **data
        }

    def _build_query(self, query_filter: Dict[str, Any]) -> Optional[Query]:
        """Build TinyDB query from filter dictionary."""
        if not query_filter:
            return None

        query = Query()
        conditions = [query[key] == value for key, value in query_filter.items()]

        # Combine conditions with AND
        result = conditions[0]
        for condition in conditions[1:]:
            result = result & condition
        return result

    def _sort_and_limit_results(self, results: List[Dict], order_by: str = None,
                               descending: bool = True, limit: int = None) -> List[Dict]:
        """Apply sorting and limiting to query results."""
        if order_by and results:
            results.sort(key=lambda x: x.get(order_by, ''), reverse=descending)

        if limit:
            results = results[:limit]

        return results

    def db_exists(self, file_path: str) -> bool:
        """Check if database file exists for given file path."""
        return os.path.exists(self._get_db_path(file_path))

    def store_data(self, file_path: str, table_name: str, data: Dict[str, Any]) -> int:
        """
        Store data in the TinyDB database for a file.

        Args:
            file_path: Path to the main file
            table_name: Name of the table to store data in
            data: Dictionary of data to store

        Returns:
            Document ID of the inserted record
        """
        self.logger.info(f"Storing data in table '{table_name}' for file: {file_path}")

        enriched_data = self._enrich_data(data, file_path)

        with self._db_context(file_path, table_name) as table:
            doc_id = table.insert(enriched_data)

        self.logger.debug(f"Stored data with ID {doc_id} in table '{table_name}'")
        return doc_id

    def load_data(self, file_path: str, table_name: str,
                  query_filter: Optional[Dict[str, Any]] = None,
                  limit: Optional[int] = None,
                  order_by: Optional[str] = None,
                  descending: bool = True) -> List[Dict[str, Any]]:
        """
        Load data from the TinyDB database for a file.

        Args:
            file_path: Path to the main file
            table_name: Name of the table to load data from
            query_filter: Optional filter criteria as key-value pairs
            limit: Optional limit on number of results
            order_by: Optional field to order results by
            descending: Whether to order in descending order (newest first)

        Returns:
            List of matching records
        """
        self.logger.debug(f"Loading data from table '{table_name}' for file: {file_path}")

        if not self.db_exists(file_path):
            self.logger.debug(f"Database does not exist: {self._get_db_path(file_path)}")
            return []

        with self._db_context(file_path, table_name) as table:
            query = self._build_query(query_filter)
            results = table.search(query) if query else table.all()

        results = self._sort_and_limit_results(results, order_by, descending, limit)
        self.logger.debug(f"Loaded {len(results)} records from table '{table_name}'")
        return results

    def get_latest_data(self, file_path: str, table_name: str,
                       query_filter: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Get the most recent record from a table.

        Args:
            file_path: Path to the main file
            table_name: Name of the table
            query_filter: Optional filter criteria

        Returns:
            Most recent record or None if no records found
        """
        results = self.load_data(file_path, table_name, query_filter, limit=1, order_by='timestamp')
        return results[0] if results else None

    def update_data(self, file_path: str, table_name: str,
                   query_filter: Dict[str, Any], updates: Dict[str, Any]) -> List[int]:
        """
        Update records matching the query filter.

        Args:
            file_path: Path to the main file
            table_name: Name of the table
            query_filter: Filter criteria to find records to update
            updates: Dictionary of fields to update

        Returns:
            List of document IDs that were updated
        """
        self.logger.debug(f"Updating data in table '{table_name}' for file: {file_path}")

        if not self.db_exists(file_path):
            return []

        with self._db_context(file_path, table_name) as table:
            query = self._build_query(query_filter)
            if query:
                doc_ids = table.update(updates, query)
            else:
                doc_ids = table.update(updates)

        self.logger.debug(f"Updated {len(doc_ids)} records in table '{table_name}'")
        return doc_ids

    def delete_data(self, file_path: str, table_name: str,
                   query_filter: Optional[Dict[str, Any]] = None) -> List[int]:
        """
        Delete records matching the query filter.

        Args:
            file_path: Path to the main file
            table_name: Name of the table
            query_filter: Optional filter criteria. If None, deletes all records

        Returns:
            List of document IDs that were deleted
        """
        self.logger.debug(f"Deleting data from table '{table_name}' for file: {file_path}")

        if not self.db_exists(file_path):
            return []

        with self._db_context(file_path, table_name) as table:
            query = self._build_query(query_filter)
            if query:
                doc_ids = table.remove(query)
            else:
                doc_ids = table.truncate()

        self.logger.debug(f"Deleted {len(doc_ids)} records from table '{table_name}'")
        return doc_ids

    def count_data(self, file_path: str, table_name: str,
                  query_filter: Optional[Dict[str, Any]] = None) -> int:
        """
        Count records matching the query filter.

        Args:
            file_path: Path to the main file
            table_name: Name of the table
            query_filter: Optional filter criteria

        Returns:
            Number of matching records
        """
        if not self.db_exists(file_path):
            return 0

        with self._db_context(file_path, table_name) as table:
            query = self._build_query(query_filter)
            if query:
                return len(table.search(query))
            else:
                return len(table.all())

    def get_table_names(self, file_path: str) -> List[str]:
        """
        Get all table names in the database.

        Args:
            file_path: Path to the main file

        Returns:
            List of table names
        """
        if not self.db_exists(file_path):
            return []

        with self._db_context(file_path) as db:
            return db.tables()

    def table_exists(self, file_path: str, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            file_path: Path to the main file
            table_name: Name of the table to check

        Returns:
            True if table exists, False otherwise
        """
        return table_name in self.get_table_names(file_path)

    def bulk_insert(self, file_path: str, table_name: str,
                   data_list: List[Dict[str, Any]]) -> List[int]:
        """
        Insert multiple records in a single transaction.

        Args:
            file_path: Path to the main file
            table_name: Name of the table
            data_list: List of dictionaries to insert

        Returns:
            List of document IDs of inserted records
        """
        self.logger.info(f"Bulk inserting {len(data_list)} records into table '{table_name}'")

        enriched_data = [self._enrich_data(data, file_path) for data in data_list]

        with self._db_context(file_path, table_name) as table:
            doc_ids = table.insert_multiple(enriched_data)

        self.logger.debug(f"Bulk inserted {len(doc_ids)} records")
        return doc_ids

    def clear_table(self, file_path: str, table_name: str) -> None:
        """
        Clear all records from a table.

        Args:
            file_path: Path to the main file
            table_name: Name of the table to clear
        """
        self.logger.debug(f"Clearing table '{table_name}' for file: {file_path}")

        if not self.db_exists(file_path):
            return

        with self._db_context(file_path, table_name) as table:
            table.truncate()

        self.logger.debug(f"Cleared table '{table_name}'")

    def drop_table(self, file_path: str, table_name: str) -> None:
        """
        Drop a table entirely.

        Args:
            file_path: Path to the main file
            table_name: Name of the table to drop
        """
        self.logger.debug(f"Dropping table '{table_name}' for file: {file_path}")

        if not self.db_exists(file_path):
            return

        with self._db_context(file_path) as db:
            db.drop_table(table_name)

        self.logger.debug(f"Dropped table '{table_name}'")

    def get_database_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get comprehensive information about the database.

        Args:
            file_path: Path to the main file

        Returns:
            Dictionary containing database statistics and information
        """
        if not self.db_exists(file_path):
            return {
                'exists': False,
                'path': self._get_db_path(file_path),
                'tables': [],
                'total_records': 0
            }

        tables = self.get_table_names(file_path)
        table_info = {}
        total_records = 0

        for table_name in tables:
            record_count = self.count_data(file_path, table_name)
            table_info[table_name] = record_count
            total_records += record_count

        return {
            'exists': True,
            'path': self._get_db_path(file_path),
            'tables': tables,
            'table_record_counts': table_info,
            'total_records': total_records,
            'file_size_bytes': os.path.getsize(self._get_db_path(file_path)) if self.db_exists(file_path) else 0
        }

    def backup_database(self, file_path: str, backup_path: str) -> bool:
        """
        Create a backup of the database.

        Args:
            file_path: Path to the main file
            backup_path: Path where to save the backup

        Returns:
            True if backup was successful, False otherwise
        """
        try:
            db_path = self._get_db_path(file_path)
            if not os.path.exists(db_path):
                self.logger.error(f"Database does not exist: {db_path}")
                return False

            # Ensure backup directory exists
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)

            # Copy database file
            import shutil
            shutil.copy2(db_path, backup_path)

            self.logger.info(f"Database backed up: {db_path} -> {backup_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to backup database: {e}")
            return False

    def restore_database(self, file_path: str, backup_path: str) -> bool:
        """
        Restore database from a backup.

        Args:
            file_path: Path to the main file
            backup_path: Path to the backup file

        Returns:
            True if restore was successful, False otherwise
        """
        try:
            if not os.path.exists(backup_path):
                self.logger.error(f"Backup file does not exist: {backup_path}")
                return False

            db_path = self._get_db_path(file_path)

            # Ensure database directory exists
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

            # Copy backup to database location
            import shutil
            shutil.copy2(backup_path, db_path)

            self.logger.info(f"Database restored: {backup_path} -> {db_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to restore database: {e}")
            return False

    def shutdown(self):
        """Close all cached database connections"""
        with self._cache_lock:
            for db in self._db_cache.values():
                try:
                    db.close()
                except Exception as e:
                    self.logger.error(f"Error shutting down database: {e}")
            self._db_cache.clear()
