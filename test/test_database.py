#!/usr/bin/env python3
"""
Compact efficient tests for DatabaseManager with maximum code reuse.
Run with: python test_database.py
"""

import os
import sys
import tempfile
import shutil
import json
import datetime
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path to import rome.database
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestDatabaseManager:
    """Test suite for DatabaseManager with maximum code reuse."""

    def setup_method(self):
        """Setup for each test - creates temp directory and manager."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.py")

        def get_db_path(file_path):
            return os.path.join(self.temp_dir, f"{os.path.basename(file_path)}.db")

        # Mock the config and logger modules
        with patch('rome.database.get_logger') as mock_get_logger, \
             patch('rome.database.set_attributes_from_config') as mock_set_attrs:

            # Setup logger mock
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            # Setup config mock to set default values
            def mock_set_attributes(obj, attrs):
                # Set default values for required attributes
                obj.lock_timeout = 5.0
                obj.max_retries = 3
                obj.retry_delay = 0.1

            mock_set_attrs.side_effect = mock_set_attributes

            # Import and create DatabaseManager
            from rome.database import DatabaseManager

            config = {
                'lock_timeout': 5.0,
                'max_retries': 3,
                'retry_delay': 0.1
            }

            self.manager = DatabaseManager(get_db_path, config)

    def teardown_method(self):
        """Cleanup after each test."""
        try:
            self.manager.shutdown()
        except:
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_basic_operations(self):
        """Test core CRUD operations."""
        # Test data
        data = {"key": "value", "number": 42}
        table = "test_table"

        # Store data
        doc_id = self.manager.store_data(self.test_file, table, data)
        assert doc_id > 0

        # Load data
        results = self.manager.load_data(self.test_file, table)
        assert len(results) == 1
        assert results[0]["key"] == "value"
        assert results[0]["number"] == 42
        assert "timestamp" in results[0]
        assert "file_path" in results[0]

        # Get latest
        latest = self.manager.get_latest_data(self.test_file, table)
        assert latest["key"] == "value"

        # Count
        count = self.manager.count_data(self.test_file, table)
        assert count == 1

        print("✓ Basic operations test passed")

    def test_query_operations(self):
        """Test querying with filters, sorting, and limits."""
        table = "query_test"

        # Insert test data
        test_data = [
            {"type": "A", "value": 10},
            {"type": "B", "value": 20},
            {"type": "A", "value": 30}
        ]

        for i, data in enumerate(test_data):
            self.manager.store_data(self.test_file, table, data)

        # Test basic loading (all records)
        all_results = self.manager.load_data(self.test_file, table)
        assert len(all_results) == 3

        # Test limit
        limited_results = self.manager.load_data(self.test_file, table, limit=2)
        assert len(limited_results) == 2

        # Test filter
        type_a_results = self.manager.load_data(self.test_file, table, query_filter={"type": "A"})
        assert len(type_a_results) == 2
        for result in type_a_results:
            assert result["type"] == "A"

        # Test latest with filter
        latest_a = self.manager.get_latest_data(self.test_file, table, query_filter={"type": "A"})
        assert latest_a is not None
        assert latest_a["type"] == "A"

        print("✓ Query operations test passed")

    def test_update_and_delete(self):
        """Test update and delete operations."""
        table = "mod_test"

        # Insert data
        data1 = {"status": "pending", "value": 100}
        data2 = {"status": "active", "value": 200}

        doc_id1 = self.manager.store_data(self.test_file, table, data1)
        doc_id2 = self.manager.store_data(self.test_file, table, data2)

        assert doc_id1 > 0
        assert doc_id2 > 0

        # Verify insertion
        count_before = self.manager.count_data(self.test_file, table)
        assert count_before == 2

        # Test update
        updated_ids = self.manager.update_data(
            self.test_file, table,
            {"status": "pending"},
            {"status": "completed"}
        )

        # Verify update worked
        results = self.manager.load_data(self.test_file, table, query_filter={"status": "completed"})
        assert len(results) >= 1

        # Test delete with filter
        deleted_ids = self.manager.delete_data(self.test_file, table, {"status": "completed"})

        # Verify partial deletion
        count_after_partial = self.manager.count_data(self.test_file, table)
        assert count_after_partial < count_before

        # Test delete all (no filter)
        remaining_deleted = self.manager.delete_data(self.test_file, table)

        # Verify complete deletion
        count_final = self.manager.count_data(self.test_file, table)
        assert count_final == 0

        print("✓ Update and delete test passed")

    def test_bulk_operations(self):
        """Test bulk insert and table management."""
        table = "bulk_test"

        # Bulk insert
        bulk_data = [{"item": f"item_{i}", "value": i} for i in range(5)]
        doc_ids = self.manager.bulk_insert(self.test_file, table, bulk_data)
        assert len(doc_ids) == 5

        # Verify bulk insert
        count = self.manager.count_data(self.test_file, table)
        assert count == 5

        # Verify data integrity
        results = self.manager.load_data(self.test_file, table)
        assert len(results) == 5

        # Check that all items are present
        items = [result["item"] for result in results]
        for i in range(5):
            assert f"item_{i}" in items

        # Clear table
        self.manager.clear_table(self.test_file, table)
        count = self.manager.count_data(self.test_file, table)
        assert count == 0

        print("✓ Bulk operations test passed")

    def test_table_management(self):
        """Test table existence and listing."""
        # Initially no database
        assert not self.manager.db_exists("nonexistent.py")

        # Create tables
        table1, table2 = "table1", "table2"
        self.manager.store_data(self.test_file, table1, {"data": 1})
        self.manager.store_data(self.test_file, table2, {"data": 2})

        # Check database exists now
        assert self.manager.db_exists(self.test_file)

        # Check table operations
        tables = self.manager.get_table_names(self.test_file)
        assert len(tables) >= 2  # Should have at least our two tables
        assert table1 in tables
        assert table2 in tables

        # Test table existence checks
        exists1 = self.manager.table_exists(self.test_file, table1)
        exists2 = self.manager.table_exists(self.test_file, table2)
        exists_fake = self.manager.table_exists(self.test_file, "nonexistent_table")

        assert exists1 is True
        assert exists2 is True
        assert exists_fake is False

        # Test drop table
        self.manager.drop_table(self.test_file, table1)
        assert not self.manager.table_exists(self.test_file, table1)
        assert self.manager.table_exists(self.test_file, table2)  # table2 should still exist

        print("✓ Table management test passed")

    def test_database_info_and_backup(self):
        """Test database information and backup/restore."""
        # Test non-existent database
        info = self.manager.get_database_info("nonexistent.py")
        assert not info["exists"]
        assert info["total_records"] == 0
        assert info["tables"] == []

        # Create some data
        self.manager.store_data(self.test_file, "table1", {"data": 1})
        self.manager.store_data(self.test_file, "table1", {"data": 2})
        self.manager.store_data(self.test_file, "table2", {"data": 3})

        # Test database info for existing database
        info = self.manager.get_database_info(self.test_file)
        assert info["exists"]
        assert info["total_records"] == 3
        assert "table1" in info["tables"]
        assert "table2" in info["tables"]
        assert "table_record_counts" in info
        assert info["table_record_counts"]["table1"] == 2
        assert info["table_record_counts"]["table2"] == 1

        # Test backup
        db_path = self.manager._get_db_path(self.test_file)
        assert os.path.exists(db_path)

        backup_path = os.path.join(self.temp_dir, "backup.db")
        success = self.manager.backup_database(self.test_file, backup_path)
        assert success
        assert os.path.exists(backup_path)

        # Test restore
        restore_file = os.path.join(self.temp_dir, "restored.py")
        success = self.manager.restore_database(restore_file, backup_path)
        assert success

        # Verify restored database
        restored_info = self.manager.get_database_info(restore_file)
        assert restored_info["exists"]
        assert restored_info["total_records"] == 3

        print("✓ Database info and backup test passed")

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        table = "edge_test"

        # Operations on non-existent database
        assert not self.manager.db_exists("nonexistent.py")
        assert self.manager.load_data("nonexistent.py", table) == []
        assert self.manager.get_latest_data("nonexistent.py", table) is None
        assert self.manager.count_data("nonexistent.py", table) == 0
        assert self.manager.update_data("nonexistent.py", table, {"key": "value"}, {"new": "data"}) == []
        assert self.manager.delete_data("nonexistent.py", table) == []
        assert self.manager.get_table_names("nonexistent.py") == []

        # Test with real data
        self.manager.store_data(self.test_file, table, {"test": "data"})

        # Empty queries and filters
        results = self.manager.load_data(self.test_file, table, {})
        assert len(results) >= 1  # At least our test data

        # Test load with None filter (should be treated as empty filter)
        results_none = self.manager.load_data(self.test_file, table, None)
        assert len(results_none) >= 1

        # Invalid backup/restore paths
        assert not self.manager.backup_database("nonexistent.py", "/tmp/backup_test")
        assert not self.manager.restore_database(self.test_file, "nonexistent_backup.db")

        # Test with empty data
        empty_data = {}
        doc_id = self.manager.store_data(self.test_file, table, empty_data)
        assert doc_id > 0

        print("✓ Edge cases test passed")

    def test_data_enrichment(self):
        """Test that data is properly enriched with metadata."""
        table = "enrichment_test"
        original_data = {"user_field": "user_value", "number": 123}

        # Store data
        doc_id = self.manager.store_data(self.test_file, table, original_data)
        assert doc_id > 0

        # Retrieve and check enrichment
        results = self.manager.load_data(self.test_file, table)
        assert len(results) >= 1

        result = results[0]

        # Check original data preserved
        assert result["user_field"] == "user_value"
        assert result["number"] == 123

        # Check metadata added
        assert "timestamp" in result
        assert "file_path" in result
        assert result["file_path"] == self.test_file

        # Verify timestamp format (ISO format)
        timestamp = result["timestamp"]
        try:
            parsed_timestamp = datetime.datetime.fromisoformat(timestamp)
            timestamp_valid = True

            # Verify timestamp is recent (within last minute)
            now = datetime.datetime.now()
            time_diff = abs((now - parsed_timestamp).total_seconds())
            assert time_diff < 60  # Should be very recent

        except ValueError:
            timestamp_valid = False

        assert timestamp_valid, f"Invalid timestamp format: {timestamp}"

        print("✓ Data enrichment test passed")

    def test_sorting_and_ordering(self):
        """Test sorting and ordering functionality."""
        table = "sort_test"

        # Insert data with different timestamps
        import time
        data_items = [
            {"name": "first", "value": 10},
            {"name": "second", "value": 20},
            {"name": "third", "value": 30}
        ]

        for item in data_items:
            self.manager.store_data(self.test_file, table, item)
            time.sleep(0.01)  # Small delay to ensure different timestamps

        # Test default ordering (by timestamp, descending)
        results = self.manager.load_data(self.test_file, table, order_by="timestamp")
        assert len(results) == 3

        # Verify timestamps are in descending order
        timestamps = [result["timestamp"] for result in results]
        for i in range(len(timestamps) - 1):
            assert timestamps[i] >= timestamps[i + 1]

        # Test ascending order by value
        results_asc = self.manager.load_data(self.test_file, table, order_by="value", descending=False)
        values = [result["value"] for result in results_asc]
        assert values == [10, 20, 30]  # Should be in ascending order

        # Test descending order by value
        results_desc = self.manager.load_data(self.test_file, table, order_by="value", descending=True)
        values_desc = [result["value"] for result in results_desc]
        assert values_desc == [30, 20, 10]  # Should be in descending order

        print("✓ Sorting and ordering test passed")


def run_tests():
    """Run all tests with simple test runner."""
    import traceback

    # Check for required dependencies
    try:
        import tinydb
        import portalocker
        print("✓ TinyDB and portalocker available")
    except ImportError as e:
        print(f"✗ Missing dependencies: {e}")
        print("Install with: pip install tinydb portalocker")
        return False

    test_class = TestDatabaseManager()
    test_methods = [method for method in dir(test_class) if method.startswith('test_')]

    passed = 0
    failed = 0

    # Sort test methods for consistent execution order
    test_methods.sort()

    print(f"Running {len(test_methods)} tests...")
    print("=" * 60)

    for test_method in test_methods:
        try:
            print(f"Running {test_method}...", end=" ")

            # Setup
            test_class.setup_method()

            # Run test
            getattr(test_class, test_method)()

            # Teardown
            test_class.teardown_method()

            passed += 1

        except Exception as e:
            print(f"✗ FAILED: {str(e)}")
            if "--verbose" in sys.argv or "-v" in sys.argv:
                print("Full traceback:")
                traceback.print_exc()
                print("-" * 40)
            failed += 1

            # Still try to teardown
            try:
                test_class.teardown_method()
            except:
                pass

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")

    if failed > 0:
        print(f"\nSome tests failed. Run with --verbose or -v flag to see full error details")
        return False
    else:
        print("All tests passed! 🎉")
        return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
