"""Tests for rome.database — DatabaseManager, LockingConfig, and locked file helpers."""
import json
import os
import sys
import tempfile
import time
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import portalocker

import rome.database as database_module
from rome.database import (
    DatabaseManager,
    LockingConfig,
    TimeoutLockedJSONStorage,
    ensure_dir,
    get_locking_config,
    locked_file_operation,
    locked_json_operation,
    set_locking_config,
)

DB_CONFIG = {"lock_timeout": 2.0, "max_retries": 3, "retry_delay": 0.01}


def _make_manager(tmpdir):
    """Build a DatabaseManager whose dbs live inside tmpdir."""
    def get_db_path(file_path):
        safe = file_path.replace(os.sep, "_").lstrip("_")
        return os.path.join(tmpdir, f"{safe}.json")
    return DatabaseManager(get_db_path, DB_CONFIG)


class TestLockingConfig(unittest.TestCase):
    def test_config_attributes_applied(self):
        cfg = LockingConfig({"lock_timeout": 1.5, "max_retries": 7, "retry_delay": 0.02})
        self.assertEqual(cfg.lock_timeout, 1.5)
        self.assertEqual(cfg.max_retries, 7)
        self.assertEqual(cfg.retry_delay, 0.02)

    def test_with_retry_returns_value_on_success(self):
        cfg = LockingConfig(DB_CONFIG)
        self.assertEqual(cfg.with_retry(lambda: 42), 42)

    def test_with_retry_retries_on_lock_exception(self):
        cfg = LockingConfig({"lock_timeout": 1.0, "max_retries": 3, "retry_delay": 0.001})
        calls = {"n": 0}

        def op():
            calls["n"] += 1
            if calls["n"] < 3:
                raise portalocker.LockException("busy")
            return "ok"

        self.assertEqual(cfg.with_retry(op), "ok")
        self.assertEqual(calls["n"], 3)

    def test_with_retry_raises_after_exhaustion(self):
        cfg = LockingConfig({"lock_timeout": 1.0, "max_retries": 2, "retry_delay": 0.001})

        def op():
            raise TimeoutError("nope")

        with self.assertRaises(TimeoutError):
            cfg.with_retry(op)

    def test_with_retry_reraises_other_exceptions(self):
        cfg = LockingConfig(DB_CONFIG)

        def op():
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            cfg.with_retry(op)


class TestGlobalLockingConfig(unittest.TestCase):
    def setUp(self):
        database_module._locking_config = None

    def tearDown(self):
        database_module._locking_config = None

    def test_get_without_set_raises(self):
        with self.assertRaises(RuntimeError):
            get_locking_config()

    def test_set_and_get_roundtrip(self):
        cfg = LockingConfig(DB_CONFIG)
        set_locking_config(cfg)
        self.assertIs(get_locking_config(), cfg)


class TestEnsureDir(unittest.TestCase):
    def test_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "a", "b", "c", "file.json")
            ensure_dir(target)
            self.assertTrue(os.path.isdir(os.path.dirname(target)))

    def test_idempotent_on_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "file.json")
            ensure_dir(target)
            ensure_dir(target)  # should not raise


class TestLockedFileOperations(unittest.TestCase):
    def setUp(self):
        set_locking_config(LockingConfig(DB_CONFIG))

    def tearDown(self):
        database_module._locking_config = None

    def test_locked_file_operation_writes_and_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "f.txt")
            with locked_file_operation(path, mode="w") as f:
                f.write("hello")
            with open(path) as f:
                self.assertEqual(f.read(), "hello")

    def test_locked_json_operation_creates_and_mutates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.json")
            with locked_json_operation(path, default_data={"count": 0}) as data:
                data["count"] = 5
            with open(path) as f:
                self.assertEqual(json.load(f), {"count": 5})

            with locked_json_operation(path) as data:
                self.assertEqual(data["count"], 5)
                data["count"] += 1
            with open(path) as f:
                self.assertEqual(json.load(f), {"count": 6})

    def test_locked_json_operation_recovers_from_corrupt_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            with open(path, "w") as f:
                f.write("{not valid json")
            with locked_json_operation(path, default_data={"fresh": True}) as data:
                self.assertEqual(data, {"fresh": True})
                data["fresh"] = False
            with open(path) as f:
                self.assertEqual(json.load(f), {"fresh": False})


class TestTimeoutLockedJSONStorage(unittest.TestCase):
    def setUp(self):
        set_locking_config(LockingConfig(DB_CONFIG))

    def tearDown(self):
        database_module._locking_config = None

    def test_read_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "missing.json")
            storage = TimeoutLockedJSONStorage(path, lock_timeout=1.0)
            try:
                self.assertEqual(storage.read(), {})
            finally:
                storage.close()

    def test_write_then_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "store.json")
            storage = TimeoutLockedJSONStorage(path, lock_timeout=1.0)
            try:
                storage.write({"a": [1, 2, 3]})
                self.assertEqual(storage.read(), {"a": [1, 2, 3]})
            finally:
                storage.close()


class TestDatabaseManagerCrud(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name
        self.mgr = _make_manager(self.tmpdir)
        self.file_path = "alpha"
        self.table = "items"

    def tearDown(self):
        self.mgr.shutdown()
        self._tmp.cleanup()

    def test_init_exposes_locking_values(self):
        self.assertEqual(self.mgr.lock_timeout, DB_CONFIG["lock_timeout"])
        self.assertEqual(self.mgr.max_retries, DB_CONFIG["max_retries"])
        self.assertEqual(self.mgr.retry_delay, DB_CONFIG["retry_delay"])

    def test_db_exists_false_initially(self):
        self.assertFalse(self.mgr.db_exists(self.file_path))

    def test_store_and_load_data(self):
        doc_id = self.mgr.store_data(self.file_path, self.table, {"name": "x", "value": 1})
        self.assertIsInstance(doc_id, int)

        records = self.mgr.load_data(self.file_path, self.table)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["name"], "x")
        self.assertEqual(records[0]["value"], 1)
        self.assertEqual(records[0]["file_path"], self.file_path)
        self.assertIn("timestamp", records[0])

    def test_load_with_query_filter(self):
        self.mgr.store_data(self.file_path, self.table, {"name": "a", "value": 1})
        self.mgr.store_data(self.file_path, self.table, {"name": "b", "value": 2})
        self.mgr.store_data(self.file_path, self.table, {"name": "a", "value": 3})

        matches = self.mgr.load_data(self.file_path, self.table, {"name": "a"})
        self.assertEqual(len(matches), 2)
        self.assertTrue(all(r["name"] == "a" for r in matches))

    def test_load_with_limit_and_sort(self):
        for v in [3, 1, 2]:
            self.mgr.store_data(self.file_path, self.table, {"value": v, "k": v})
            time.sleep(0.002)  # ensure distinct timestamps

        all_desc = self.mgr.load_data(self.file_path, self.table, order_by="k", descending=True)
        self.assertEqual([r["k"] for r in all_desc], [3, 2, 1])

        top_one = self.mgr.load_data(self.file_path, self.table, order_by="k", descending=True, limit=1)
        self.assertEqual(len(top_one), 1)
        self.assertEqual(top_one[0]["k"], 3)

    def test_load_missing_db_returns_empty(self):
        self.assertEqual(self.mgr.load_data("does_not_exist", self.table), [])

    def test_get_latest_data(self):
        self.mgr.store_data(self.file_path, self.table, {"n": 1})
        time.sleep(0.005)
        self.mgr.store_data(self.file_path, self.table, {"n": 2})

        latest = self.mgr.get_latest_data(self.file_path, self.table)
        self.assertIsNotNone(latest)
        self.assertEqual(latest["n"], 2)

    def test_get_latest_data_none_when_empty(self):
        self.assertIsNone(self.mgr.get_latest_data("nope", self.table))

    def test_update_data(self):
        self.mgr.store_data(self.file_path, self.table, {"name": "a", "value": 1})
        self.mgr.store_data(self.file_path, self.table, {"name": "b", "value": 2})

        updated_ids = self.mgr.update_data(self.file_path, self.table, {"name": "a"}, {"value": 99})
        self.assertEqual(len(updated_ids), 1)

        rec = self.mgr.load_data(self.file_path, self.table, {"name": "a"})[0]
        self.assertEqual(rec["value"], 99)

    def test_update_missing_db_returns_empty(self):
        self.assertEqual(self.mgr.update_data("nope", self.table, {"x": 1}, {"y": 2}), [])

    def test_delete_with_query(self):
        self.mgr.store_data(self.file_path, self.table, {"name": "a"})
        self.mgr.store_data(self.file_path, self.table, {"name": "b"})

        removed = self.mgr.delete_data(self.file_path, self.table, {"name": "a"})
        self.assertEqual(len(removed), 1)
        self.assertEqual(self.mgr.count_data(self.file_path, self.table), 1)

    def test_delete_without_query_truncates(self):
        self.mgr.store_data(self.file_path, self.table, {"name": "a"})
        self.mgr.store_data(self.file_path, self.table, {"name": "b"})

        result = self.mgr.delete_data(self.file_path, self.table)
        self.assertEqual(result, [])
        self.assertEqual(self.mgr.count_data(self.file_path, self.table), 0)

    def test_delete_missing_db_returns_empty(self):
        self.assertEqual(self.mgr.delete_data("nope", self.table), [])

    def test_count_data(self):
        self.assertEqual(self.mgr.count_data("nope", self.table), 0)
        for i in range(3):
            self.mgr.store_data(self.file_path, self.table, {"i": i})
        self.assertEqual(self.mgr.count_data(self.file_path, self.table), 3)
        self.assertEqual(self.mgr.count_data(self.file_path, self.table, {"i": 1}), 1)

    def test_get_table_names_and_table_exists(self):
        self.assertEqual(self.mgr.get_table_names("nope"), [])
        self.mgr.store_data(self.file_path, "t1", {"x": 1})
        self.mgr.store_data(self.file_path, "t2", {"y": 2})

        tables = self.mgr.get_table_names(self.file_path)
        self.assertIn("t1", tables)
        self.assertIn("t2", tables)

        self.assertTrue(self.mgr.table_exists(self.file_path, "t1"))
        self.assertFalse(self.mgr.table_exists(self.file_path, "missing"))

    def test_bulk_insert(self):
        data = [{"v": i} for i in range(5)]
        ids = self.mgr.bulk_insert(self.file_path, self.table, data)
        self.assertEqual(len(ids), 5)
        self.assertEqual(self.mgr.count_data(self.file_path, self.table), 5)

    def test_clear_table(self):
        for i in range(3):
            self.mgr.store_data(self.file_path, self.table, {"i": i})
        self.mgr.clear_table(self.file_path, self.table)
        self.assertEqual(self.mgr.count_data(self.file_path, self.table), 0)

    def test_clear_missing_db_is_noop(self):
        self.mgr.clear_table("nope", self.table)  # should not raise

    def test_drop_table(self):
        self.mgr.store_data(self.file_path, "keep", {"x": 1})
        self.mgr.store_data(self.file_path, "drop_me", {"y": 2})

        self.mgr.drop_table(self.file_path, "drop_me")
        self.assertFalse(self.mgr.table_exists(self.file_path, "drop_me"))
        self.assertTrue(self.mgr.table_exists(self.file_path, "keep"))

    def test_get_database_info_missing(self):
        info = self.mgr.get_database_info("nope")
        self.assertFalse(info["exists"])
        self.assertEqual(info["tables"], [])
        self.assertEqual(info["total_records"], 0)
        self.assertIn("path", info)

    def test_get_database_info_populated(self):
        self.mgr.store_data(self.file_path, "t1", {"x": 1})
        self.mgr.store_data(self.file_path, "t1", {"x": 2})
        self.mgr.store_data(self.file_path, "t2", {"y": 1})

        info = self.mgr.get_database_info(self.file_path)
        self.assertTrue(info["exists"])
        self.assertEqual(info["total_records"], 3)
        self.assertEqual(info["table_record_counts"]["t1"], 2)
        self.assertEqual(info["table_record_counts"]["t2"], 1)
        self.assertGreater(info["file_size_bytes"], 0)

    def test_shutdown_clears_global_locking_config(self):
        self.assertIsNotNone(database_module._locking_config)
        self.mgr.shutdown()
        self.assertIsNone(database_module._locking_config)
        # Re-initialize so tearDown's shutdown is a no-op re-run
        self.mgr = _make_manager(self.tmpdir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
