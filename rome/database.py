"""
Compact, deadlock-safe TinyDB management with maximum code reuse.
Includes unified locking operations for file and JSON handling.
"""
import datetime
import json
import os
import time
from typing import List, Dict, Any, Optional, Callable
from contextlib import contextmanager

from tinydb import TinyDB, Query
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware
import portalocker

from .config import set_attributes_from_config
from .logger import get_logger

def ensure_dir(file_path: str) -> None:
    """Thread-safe directory creation."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    except FileExistsError:
        pass


@contextmanager
def locked_file_operation(file_path: str, mode: str = 'r',
                         lock_type: int = portalocker.LOCK_EX,
                         timeout: float = 5.0, encoding: str = 'utf-8',
                         create_dirs: bool = True):
    """File operation with timeout-based locking."""
    if create_dirs:
        ensure_dir(file_path)

    with open(file_path, mode, encoding=encoding) as f:
        # Reuse the existing timeout logic pattern
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                portalocker.lock(f, lock_type | portalocker.LOCK_NB)
                break
            except portalocker.LockException:
                time.sleep(0.01)
        else:
            raise TimeoutError(f"Lock timeout on {file_path}")

        yield f


@contextmanager
def locked_json_operation(file_path: str, default_data: Optional[Dict] = None,
                         timeout: float = 5.0, create_dirs: bool = True,
                         logger=None):
    """JSON operation with timeout-based locking and atomic writes."""
    if create_dirs:
        ensure_dir(file_path)

    with open(file_path, 'a+', encoding='utf-8') as f:
        # Reuse the existing timeout logic pattern
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                portalocker.lock(f, portalocker.LOCK_EX | portalocker.LOCK_NB)
                break
            except portalocker.LockException:
                time.sleep(0.01)
        else:
            raise TimeoutError(f"Lock timeout on {file_path}")

        # Read existing content
        f.seek(0)
        content = f.read()

        data = default_data or {}
        if content.strip():
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                if logger:
                    logger.error(f"Corrupted JSON file {file_path}: {e}")
                data = default_data or {}

        yield data

        # Atomic write
        f.seek(0)
        f.truncate()
        json.dump(data, f, indent=4)


class TimeoutLockedJSONStorage(JSONStorage):
    """TinyDB storage with timeout-based file locking"""

    def __init__(self, path: str, create_dirs: bool = True, lock_timeout: float = 5.0, **kwargs):
        self.path = path
        self.lock_timeout = lock_timeout
        if create_dirs:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        super().__init__(path, **kwargs)

    def _lock_with_timeout(self, file_handle, lock_type):
        """Acquire file lock with timeout"""
        start_time = time.time()
        while time.time() - start_time < self.lock_timeout:
            try:
                portalocker.lock(file_handle, lock_type | portalocker.LOCK_NB)
                return
            except portalocker.LockException:
                time.sleep(0.01)
        raise TimeoutError(f"Lock timeout on {self.path}")

    def read(self):
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                self._lock_with_timeout(f, portalocker.LOCK_SH)
                content = f.read().strip()
                return json.loads(content) if content else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def write(self, data):
        with open(self.path, 'w', encoding='utf-8') as f:
            self._lock_with_timeout(f, portalocker.LOCK_EX)
            json.dump(data, f, indent=4)


class DatabaseManager:
    """Compact, deadlock-safe TinyDB manager with maximum code reuse"""

    def __init__(self, get_db_path_func, config: Dict = None):
        self.config = config or {}
        self.logger = get_logger()
        self._get_db_path = get_db_path_func

        # Configuration with defaults
        set_attributes_from_config(self, ['lock_timeout', 'max_retries', 'retry_delay'])

    def _timestamp(self) -> str:
        """Generate ISO timestamp"""
        return datetime.datetime.now().isoformat()

    @contextmanager
    def _db_context(self, file_path: str, table_name: str = None):
        """Database context with automatic cleanup"""
        db_path = self._get_db_path(file_path)
        storage = CachingMiddleware(lambda path: TimeoutLockedJSONStorage(path, lock_timeout=self.lock_timeout))
        db = TinyDB(db_path, storage=storage)

        try:
            yield db.table(table_name) if table_name else db
        finally:
            try:
                db.close()
            except Exception as e:
                self.logger.error(f"Error closing database: {e}")

    def _with_retry(self, operation: Callable, *args, **kwargs):
        """Execute operation with retry logic"""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except (TimeoutError, portalocker.LockException) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    self.logger.error(f"Lock timeout on attempt {attempt + 1}, retrying...")
                    time.sleep(self.retry_delay * (2 ** attempt))
                continue
            except Exception:
                raise

        raise last_exception

    def _enrich_data(self, data: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """Add metadata to data"""
        return {'timestamp': self._timestamp(), 'file_path': file_path, **data}

    def _build_query(self, query_filter: Dict[str, Any]) -> Optional[Query]:
        """Build TinyDB query from filter dict"""
        if not query_filter:
            return None

        query = Query()
        conditions = [query[key] == value for key, value in query_filter.items()]
        result = conditions[0]
        for condition in conditions[1:]:
            result = result & condition
        return result

    def _sort_and_limit(self, results: List[Dict], order_by: str = None,
                       descending: bool = True, limit: int = None) -> List[Dict]:
        """Apply sorting and limiting"""
        if order_by and results:
            results.sort(key=lambda x: x.get(order_by, ''), reverse=descending)
        return results[:limit] if limit else results

    def _safe_len(self, obj) -> int:
        """Safe length calculation"""
        return len(obj) if obj is not None else 0

    # Core database operations with unified error handling
    def db_exists(self, file_path: str) -> bool:
        """Check if database exists"""
        return os.path.exists(self._get_db_path(file_path))

    def store_data(self, file_path: str, table_name: str, data: Dict[str, Any]) -> int:
        """Store data with automatic retry"""
        def _store():
            enriched_data = self._enrich_data(data, file_path)
            with self._db_context(file_path, table_name) as table:
                return table.insert(enriched_data)

        doc_id = self._with_retry(_store)
        self.logger.debug(f"Stored data with ID {doc_id} in '{table_name}'")
        return doc_id

    def load_data(self, file_path: str, table_name: str, query_filter: Optional[Dict[str, Any]] = None,
                  limit: Optional[int] = None, order_by: Optional[str] = None,
                  descending: bool = True) -> List[Dict[str, Any]]:
        """Load data with filtering and sorting"""
        def _load():
            if not self.db_exists(file_path):
                return []

            with self._db_context(file_path, table_name) as table:
                query = self._build_query(query_filter)
                results = table.search(query) if query else table.all()

            return self._sort_and_limit(results, order_by, descending, limit)

        results = self._with_retry(_load)
        self.logger.debug(f"Loaded {len(results)} records from '{table_name}'")
        return results

    def get_latest_data(self, file_path: str, table_name: str,
                       query_filter: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Get most recent record"""
        results = self.load_data(file_path, table_name, query_filter, limit=1, order_by='timestamp')
        return results[0] if results else None

    def update_data(self, file_path: str, table_name: str, query_filter: Dict[str, Any],
                   updates: Dict[str, Any]) -> List[int]:
        """Update records"""
        def _update():
            if not self.db_exists(file_path):
                return []

            with self._db_context(file_path, table_name) as table:
                query = self._build_query(query_filter)
                doc_ids = table.update(updates, query) if query else table.update(updates)
                return doc_ids or []

        doc_ids = self._with_retry(_update)
        self.logger.debug(f"Updated {self._safe_len(doc_ids)} records in '{table_name}'")
        return doc_ids

    def delete_data(self, file_path: str, table_name: str,
                   query_filter: Optional[Dict[str, Any]] = None) -> List[int]:
        """Delete records"""
        def _delete():
            if not self.db_exists(file_path):
                return []

            with self._db_context(file_path, table_name) as table:
                query = self._build_query(query_filter)
                if query:
                    doc_ids = table.remove(query) or []
                    self.logger.debug(f"Deleted {self._safe_len(doc_ids)} records from '{table_name}'")
                    return doc_ids
                else:
                    # Handle truncate separately since it returns None
                    count = len(table.all())
                    table.truncate()
                    self.logger.debug(f"Truncated {count} records from '{table_name}'")
                    return []

        return self._with_retry(_delete)

    def count_data(self, file_path: str, table_name: str,
                  query_filter: Optional[Dict[str, Any]] = None) -> int:
        """Count records"""
        def _count():
            if not self.db_exists(file_path):
                return 0

            with self._db_context(file_path, table_name) as table:
                query = self._build_query(query_filter)
                return len(table.search(query) if query else table.all())

        return self._with_retry(_count)

    def get_table_names(self, file_path: str) -> List[str]:
        """Get all table names"""
        def _get_tables():
            if not self.db_exists(file_path):
                return []
            with self._db_context(file_path) as db:
                return db.tables()

        return self._with_retry(_get_tables)

    def table_exists(self, file_path: str, table_name: str) -> bool:
        """Check if table exists"""
        return table_name in self.get_table_names(file_path)

    def bulk_insert(self, file_path: str, table_name: str, data_list: List[Dict[str, Any]]) -> List[int]:
        """Bulk insert records"""
        def _bulk_insert():
            enriched_data = [self._enrich_data(data, file_path) for data in data_list]
            with self._db_context(file_path, table_name) as table:
                return table.insert_multiple(enriched_data)

        doc_ids = self._with_retry(_bulk_insert)
        self.logger.debug(f"Bulk inserted {len(doc_ids)} records into '{table_name}'")
        return doc_ids

    def clear_table(self, file_path: str, table_name: str) -> None:
        """Clear all records from table"""
        def _clear():
            if self.db_exists(file_path):
                with self._db_context(file_path, table_name) as table:
                    table.truncate()

        self._with_retry(_clear)
        self.logger.debug(f"Cleared table '{table_name}'")

    def drop_table(self, file_path: str, table_name: str) -> None:
        """Drop table entirely"""
        def _drop():
            if self.db_exists(file_path):
                with self._db_context(file_path) as db:
                    db.drop_table(table_name)

        self._with_retry(_drop)
        self.logger.debug(f"Dropped table '{table_name}'")

    def get_database_info(self, file_path: str) -> Dict[str, Any]:
        """Get database information"""
        if not self.db_exists(file_path):
            return {
                'exists': False,
                'path': self._get_db_path(file_path),
                'tables': [],
                'total_records': 0
            }

        def _get_info():
            tables = self.get_table_names(file_path)
            table_info = {table: self.count_data(file_path, table) for table in tables}
            return {
                'exists': True,
                'path': self._get_db_path(file_path),
                'tables': tables,
                'table_record_counts': table_info,
                'total_records': sum(table_info.values()),
                'file_size_bytes': os.path.getsize(self._get_db_path(file_path))
            }

        return self._with_retry(_get_info)

    def backup_database(self, file_path: str, backup_path: str) -> bool:
        """Backup database"""
        try:
            db_path = self._get_db_path(file_path)
            if not os.path.exists(db_path):
                self.logger.error(f"Database does not exist: {db_path}")
                return False

            os.makedirs(os.path.dirname(backup_path), exist_ok=True)

            import shutil
            shutil.copy2(db_path, backup_path)
            self.logger.info(f"Database backed up: {db_path} -> {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return False

    def restore_database(self, file_path: str, backup_path: str) -> bool:
        """Restore database from backup"""
        try:
            if not os.path.exists(backup_path):
                self.logger.error(f"Backup file does not exist: {backup_path}")
                return False

            db_path = self._get_db_path(file_path)
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

            import shutil
            shutil.copy2(backup_path, db_path)
            self.logger.info(f"Database restored: {backup_path} -> {db_path}")
            return True
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return False

    def shutdown(self):
        """Shutdown - no cached connections to close"""
        self.logger.debug("DatabaseManager shutdown complete")
