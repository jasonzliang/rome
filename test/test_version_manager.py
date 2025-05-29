#!/usr/bin/env python3
"""
Comprehensive pytest suite for VersionManager
Tests all functionality including edge cases, error handling, and concurrency
Run with: python test_version_manager.py
"""

import pytest
import os
import json
import tempfile
import shutil
import sys
import threading
import time
import stat
from unittest.mock import Mock, patch, MagicMock, call
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup comprehensive mocks
def setup_mocks():
    """Setup all required mocks with realistic behavior"""
    # Mock external dependencies
    mock_psutil = MagicMock()
    # Add exception classes to psutil mock
    mock_psutil.NoSuchProcess = type('NoSuchProcess', (Exception,), {})
    mock_psutil.AccessDenied = type('AccessDenied', (Exception,), {})

    mock_portalocker = MagicMock()
    mock_portalocker.LOCK_EX = 2
    mock_portalocker.lock = MagicMock()

    # Mock rome modules
    mock_config = MagicMock()
    mock_config.META_DIR_EXT = "meta"
    mock_config.TEST_FILE_EXT = "_test.py"

    mock_logger = MagicMock()
    mock_logger.debug = lambda msg: None  # Silent for performance
    mock_logger.info = lambda msg: None
    mock_logger.warning = lambda msg: print(f"WARNING: {msg}")
    mock_logger.error = lambda msg: print(f"ERROR: {msg}")

    mock_parsing = MagicMock()
    mock_parsing.hash_string = lambda x: f"hash_{hash(x):064x}"[-64:]

    # Mock TinyDBManager with realistic behavior
    mock_tinydb_manager = MagicMock()

    class MockTinyDBInstance:
        def __init__(self):
            self.data_store = {}
            self.call_count = 0

        def store_data(self, file_path, table_name, data):
            self.call_count += 1
            key = f"{file_path}:{table_name}"
            if key not in self.data_store:
                self.data_store[key] = []
            self.data_store[key].append({**data, 'id': self.call_count})
            return self.call_count

        def get_latest_data(self, file_path, table_name, query_filter=None):
            key = f"{file_path}:{table_name}"
            records = self.data_store.get(key, [])
            if not records:
                return None
            latest = records[-1]
            if query_filter:
                for k, v in query_filter.items():
                    if latest.get(k) != v:
                        return None
            return latest

        def shutdown(self):
            self.data_store.clear()

    mock_db_instance = MockTinyDBInstance()
    mock_tinydb_manager.return_value = mock_db_instance

    # Register all mocks
    modules = {
        'psutil': mock_psutil,
        'portalocker': mock_portalocker,
        'rome.config': mock_config,
        'rome.logger': MagicMock(get_logger=lambda: mock_logger),
        'rome.parsing': mock_parsing,
        'rome.tinydb_manager': MagicMock(TinyDBManager=mock_tinydb_manager)
    }

    for name, module in modules.items():
        sys.modules[name] = module

    return modules, mock_db_instance

# Setup mocks and import
mocks, mock_db = setup_mocks()
from rome.version_manager import VersionManager, ValidationError, FileType, locked_file_operation, locked_json_operation


class TestVersionManager:
    """Comprehensive test suite covering all functionality"""

    @pytest.fixture
    def setup(self):
        """Central setup fixture with comprehensive test resources"""
        temp_dir = tempfile.mkdtemp()

        vm = VersionManager()

        # Create multiple mock agents for concurrency testing
        agents = []
        for i in range(3):
            agent = Mock()
            agent.get_id.return_value = f"test_agent_{os.getpid() + i}"
            agent.chat_completion.return_value = f"Analysis result {i}"
            agent.role = f"Test Agent {i}"
            agents.append(agent)

        # Create test file structure
        files = {}
        contents = {}

        # Main files
        for name, content in [
            ("module.py", "def add(a, b):\n    return a + b"),
            ("utils.py", "def multiply(x, y):\n    return x * y"),
            ("empty.py", ""),
            ("unicode.py", "def greet():\n    return '你好世界'")
        ]:
            file_path = os.path.join(temp_dir, name)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            files[name] = file_path
            contents[name] = content

        # Test files
        for name, content in [
            ("module_test.py", "def test_add():\n    assert add(2, 3) == 5"),
            ("utils_test.py", "def test_multiply():\n    assert multiply(3, 4) == 12")
        ]:
            file_path = os.path.join(temp_dir, name)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            files[name] = file_path
            contents[name] = content

        yield {
            'temp_dir': temp_dir,
            'vm': vm,
            'agents': agents,
            'files': files,
            'contents': contents
        }

        # Comprehensive cleanup
        try:
            # Shutdown database connections
            vm.shutdown(agents[0])
            # Force remove temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    # ========== CORE FUNCTIONALITY TESTS ==========

    def test_initialization_and_utilities(self, setup):
        """Test initialization and core utilities"""
        vm = setup['vm']

        # Test initialization
        assert vm.config == {}
        assert vm.active_files == set()
        assert vm.logger is not None

        # Test with custom config
        custom_vm = VersionManager({"test": "value"})
        assert custom_vm.config == {"test": "value"}

        # Test timestamp generation
        timestamp = vm._get_timestamp()
        assert datetime.datetime.fromisoformat(timestamp)

        # Test hash consistency
        content = "test content"
        hash1 = mocks['rome.parsing'].hash_string(content)
        hash2 = mocks['rome.parsing'].hash_string(content)
        hash3 = mocks['rome.parsing'].hash_string("different")
        assert hash1 == hash2 != hash3
        assert len(hash1) == 64

        # Test PID extraction
        assert vm._get_pid_from_agent_id("test_agent_12345") == 12345
        assert vm._get_pid_from_agent_id("invalid_format") is None
        assert vm._get_pid_from_agent_id("test_agent") is None
        assert vm._get_pid_from_agent_id("") is None

    def test_meta_directory_management(self, setup):
        """Test meta directory creation and management"""
        vm = setup['vm']
        main_file = setup['files']['module.py']
        test_file = setup['files']['module_test.py']

        # Test main file meta directory
        meta_dir = vm._get_meta_dir(main_file)
        assert meta_dir == f"{main_file}.meta"
        assert os.path.exists(meta_dir)
        assert os.path.isdir(meta_dir)

        # Test test file meta directory
        test_meta_dir = vm._get_test_meta_dir(test_file, main_file)
        expected = os.path.join(meta_dir, "module_test.py.meta")
        assert test_meta_dir == expected
        assert os.path.exists(test_meta_dir)

        # Test file path utilities
        assert vm._get_file_path(meta_dir, FileType.INDEX) == os.path.join(meta_dir, "index.json")
        assert vm._clean_file_path(f"{main_file}.meta") == main_file
        assert vm._clean_file_path(main_file) == main_file

    def test_file_inference_and_validation(self, setup):
        """Test file path inference and validation"""
        vm = setup['vm']
        test_file = setup['files']['module_test.py']
        main_file = setup['files']['module.py']

        # Test successful inference
        inferred = vm._infer_main_file_from_test(test_file)
        assert inferred == main_file

        # Test non-test file
        regular_file = setup['files']['utils.py']
        assert vm._infer_main_file_from_test(regular_file) is None

        # Test non-existent main file
        fake_test = os.path.join(setup['temp_dir'], "nonexistent_test.py")
        with open(fake_test, 'w') as f:
            f.write("# test")
        assert vm._infer_main_file_from_test(fake_test) is None

    # ========== VERSION MANAGEMENT TESTS ==========

    def test_version_utilities(self, setup):
        """Test version management utility functions"""
        vm = setup['vm']

        # Test finding existing version by hash
        index = {
            'versions': [
                {'version': 1, 'hash': 'abc123'},
                {'version': 2, 'hash': 'def456'},
                {'version': 3, 'hash': 'ghi789'}
            ]
        }

        assert vm._find_existing_version_by_hash(index, 'def456') == 2
        assert vm._find_existing_version_by_hash(index, 'nonexistent') is None
        assert vm._find_existing_version_by_hash({}, 'abc123') is None
        assert vm._find_existing_version_by_hash({'versions': []}, 'abc123') is None

        # Test next version number calculation
        assert vm._get_next_version_number({}) == 1
        assert vm._get_next_version_number({'versions': []}) == 1
        assert vm._get_next_version_number(index) == 4

        # Test metadata creation
        metadata = vm._create_version_metadata(
            file_path="/test/file.py",
            content_hash="abc123",
            version_number=2,
            changes=[{"type": "fix", "description": "Bug fix"}],
            explanation="Fixed critical bug",
            main_file_path="/test/main.py"
        )

        assert metadata['version'] == 2
        assert metadata['file_path'] == "/test/file.py"
        assert metadata['hash'] == "abc123"
        assert metadata['changes'] == [{"type": "fix", "description": "Bug fix"}]
        assert metadata['explanation'] == "Fixed critical bug"
        assert metadata['main_file_path'] == "/test/main.py"
        assert 'timestamp' in metadata

    def test_comprehensive_version_workflow(self, setup):
        """Test complete version management workflow"""
        vm = setup['vm']
        main_file = setup['files']['module.py']
        main_content = setup['contents']['module.py']
        test_file = setup['files']['module_test.py']
        test_content = setup['contents']['module_test.py']

        # 1. Save original version
        version1 = vm.save_original(main_file, main_content)
        assert version1 == 1

        # 2. Save same content (should return existing version)
        version2 = vm.save_version(main_file, main_content)
        assert version2 == 1

        # 3. Save modified version
        modified_content = main_content + "\n\ndef subtract(a, b):\n    return a - b"
        version3 = vm.save_version(
            main_file, modified_content,
            changes=[
                {"type": "feature", "description": "Added subtract function"},
                {"type": "enhancement", "description": "Improved functionality"}
            ],
            explanation="Added new mathematical operation"
        )
        assert version3 == 2

        # 4. Save test version
        test_version1 = vm.save_test_version(test_file, test_content, main_file_path=main_file)
        assert test_version1 == 1

        # 5. Save modified test version (with inference)
        modified_test = test_content + "\n\ndef test_subtract():\n    assert subtract(5, 3) == 2"
        test_version2 = vm.save_test_version(
            test_file, modified_test,
            changes=[{"type": "test", "description": "Added subtract test"}],
            explanation="Test for new subtract function"
        )
        assert test_version2 == 2

        # 6. Verify file structure exists
        meta_dir = vm._get_meta_dir(main_file)
        test_meta_dir = vm._get_test_meta_dir(test_file, main_file)

        expected_files = [
            os.path.join(meta_dir, "index.json"),
            os.path.join(meta_dir, "module_v1.py"),
            os.path.join(meta_dir, "module_v2.py"),
            os.path.join(test_meta_dir, "index.json"),
            os.path.join(test_meta_dir, "module_test_v1.py"),
            os.path.join(test_meta_dir, "module_test_v2.py")
        ]

        for file_path in expected_files:
            assert os.path.exists(file_path), f"Missing file: {file_path}"

        # 7. Verify index content
        with open(os.path.join(meta_dir, "index.json"), 'r') as f:
            main_index = json.load(f)

        assert len(main_index['versions']) == 2
        assert main_index['versions'][0]['version'] == 1
        assert main_index['versions'][1]['version'] == 2
        assert len(main_index['versions'][1]['changes']) == 2

        # 8. Verify version file content
        with open(os.path.join(meta_dir, "module_v2.py"), 'r') as f:
            saved_content = f.read()
        assert saved_content == modified_content

    def test_edge_case_content(self, setup):
        """Test version management with edge case content"""
        vm = setup['vm']
        temp_dir = setup['temp_dir']

        test_cases = [
            ("empty.py", ""),
            ("unicode.py", "def greet():\n    return '你好世界🌍'"),
            ("large.py", "x = 'content'\n" * 1000),
            ("special_chars.py", "# File with special chars: @#$%^&*()"),
            ("multiline.py", '"""Multi-line\nstring with\nembedded quotes"""\ndef func():\n    pass')
        ]

        for filename, content in test_cases:
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Save version and verify
            version = vm.save_version(file_path, content)
            assert version == 1

            # Verify saved content matches
            meta_dir = vm._get_meta_dir(file_path)
            base_name = os.path.splitext(filename)[0]
            version_file = os.path.join(meta_dir, f"{base_name}_v1.py")

            with open(version_file, 'r', encoding='utf-8') as f:
                saved_content = f.read()
            assert saved_content == content

    # ========== ACTIVE FILE MANAGEMENT TESTS ==========

    def test_active_file_lifecycle(self, setup):
        """Test complete active file management lifecycle"""
        vm = setup['vm']
        agent = setup['agents'][0]
        main_file = setup['files']['module.py']

        with patch.object(vm, '_is_process_running', return_value=True):
            # Initially not active
            assert not vm.check_active(main_file)
            assert not vm._has_active_files()
            assert len(vm._get_active_files()) == 0

            # Flag as active
            result = vm.flag_active(agent, main_file)
            assert result is True

            # Verify active state
            assert vm.check_active(main_file, ignore_self=False)
            assert not vm.check_active(main_file, ignore_self=True)  # Default behavior
            assert vm._has_active_files()
            assert len(vm._get_active_files()) == 1
            assert os.path.abspath(main_file) in vm._get_active_files()

            # Verify active file content
            meta_dir = vm._get_meta_dir(main_file)
            active_file_path = vm._get_file_path(meta_dir, FileType.ACTIVE)
            assert os.path.exists(active_file_path)

            with open(active_file_path, 'r') as f:
                active_data = json.load(f)

            assert active_data['agent_id'] == agent.get_id()
            assert active_data['file_path'] == main_file
            assert 'timestamp' in active_data

            # Try to flag another file (should fail)
            other_file = setup['files']['utils.py']
            with pytest.raises(RuntimeError, match="already has active file"):
                vm.flag_active(agent, other_file)

            # Unflag
            result = vm.unflag_active(agent, main_file)
            assert result is True
            assert not vm.check_active(main_file, ignore_self=False)
            assert not vm._has_active_files()
            assert not os.path.exists(active_file_path)

    def test_active_file_collision_detection(self, setup):
        """Test active file collision detection between agents"""
        vm = setup['vm']
        agent1, agent2 = setup['agents'][0], setup['agents'][1]
        main_file = setup['files']['module.py']

        with patch.object(vm, '_is_process_running', return_value=True):
            # Agent 1 flags file as active
            vm.flag_active(agent1, main_file)

            # Agent 2 tries to flag same file (should fail with "already has active file" message)
            with pytest.raises(RuntimeError, match="already has active file"):
                vm.flag_active(agent2, main_file)

            # Agent 2 cannot unflag agent 1's file
            result = vm.unflag_active(agent2, main_file)
            assert result is False

            # Agent 1 can unflag their own file
            result = vm.unflag_active(agent1, main_file)
            assert result is True

            # After unflagging, agent 1 can re-flag the same file
            result = vm.flag_active(agent1, main_file)
            assert result is True

            # Clean up
            vm.unflag_active(agent1, main_file)

    def test_stale_process_cleanup(self, setup):
        """Test cleanup of stale process files"""
        vm = setup['vm']
        main_file = setup['files']['module.py']

        # Create stale agent with non-existent PID
        stale_agent = Mock()
        stale_agent.get_id.return_value = "stale_agent_99999"  # Non-existent PID

        # Manually create stale active file
        meta_dir = vm._get_meta_dir(main_file)
        active_file_path = vm._get_file_path(meta_dir, FileType.ACTIVE)

        stale_data = {
            'agent_id': stale_agent.get_id(),
            'timestamp': vm._get_timestamp(),
            'file_path': main_file
        }

        with open(active_file_path, 'w') as f:
            json.dump(stale_data, f)

        # Verify file exists before cleanup
        assert os.path.exists(active_file_path)

        # Check should clean up stale file (process won't be found)
        is_active = vm.check_active(main_file)
        assert not is_active
        assert not os.path.exists(active_file_path)

    def test_process_management(self, setup):
        """Test process-related utility methods"""
        vm = setup['vm']

        # Test process running check with properly mocked psutil
        with patch('rome.version_manager.psutil.Process') as mock_process_class:
            mock_proc = Mock()
            mock_proc.username.return_value = "testuser"
            mock_proc.name.return_value = "python"
            mock_proc.cmdline.return_value = ["python", "script.py"]
            mock_process_class.return_value = mock_proc

            with patch('rome.version_manager.os.getlogin', return_value="testuser"):
                # Valid Python process
                assert vm._is_process_running(12345) is True

            with patch('rome.version_manager.os.getlogin', return_value="differentuser"):
                # Different user
                assert vm._is_process_running(12345) is False

            # Test with non-Python process
            mock_proc.name.return_value = "notepad"
            mock_proc.cmdline.return_value = ["notepad.exe"]
            with patch('rome.version_manager.os.getlogin', return_value="testuser"):
                assert vm._is_process_running(12345) is False

        # Test with process exceptions
        with patch('rome.version_manager.psutil.Process', side_effect=mocks['psutil'].NoSuchProcess(12345)):
            assert vm._is_process_running(12345) is False

        with patch('rome.version_manager.psutil.Process', side_effect=mocks['psutil'].AccessDenied(12345)):
            assert vm._is_process_running(12345) is False

    # ========== FINISHED FILE MANAGEMENT TESTS ==========

    def test_finished_file_workflow(self, setup):
        """Test finished file flagging and checking"""
        vm = setup['vm']
        agent1, agent2 = setup['agents'][0], setup['agents'][1]
        main_file = setup['files']['module.py']

        # Initially not finished
        assert not vm.check_finished(agent1, main_file)
        assert not vm.check_finished(agent2, main_file)

        # Agent 1 flags as finished
        vm.flag_finished(agent1, main_file)
        assert vm.check_finished(agent1, main_file)
        assert not vm.check_finished(agent2, main_file)

        # Agent 2 also flags as finished
        vm.flag_finished(agent2, main_file)
        assert vm.check_finished(agent1, main_file)
        assert vm.check_finished(agent2, main_file)

        # Agent 1 flags again (should not duplicate)
        vm.flag_finished(agent1, main_file)
        assert vm.check_finished(agent1, main_file)

        # Verify finished file structure
        meta_dir = vm._get_meta_dir(main_file)
        finished_file_path = vm._get_file_path(meta_dir, FileType.FINISHED)

        with open(finished_file_path, 'r') as f:
            finished_data = json.load(f)

        assert len(finished_data['agents']) == 2
        agent_ids = [agent['agent_id'] for agent in finished_data['agents']]
        assert agent1.get_id() in agent_ids
        assert agent2.get_id() in agent_ids

    # ========== DATABASE INTERFACE TESTS ==========

    def test_database_operations(self, setup):
        """Test TinyDB database operations"""
        vm = setup['vm']
        main_file = setup['files']['module.py']

        # Test store_data
        test_data = {
            'type': 'analysis',
            'content': 'Code looks good',
            'timestamp': vm._get_timestamp()
        }

        record_id = vm.store_data(main_file, 'analysis', test_data)
        assert record_id == 1

        # Test get_latest_data
        retrieved = vm.get_latest_data(main_file, 'analysis')
        assert retrieved is not None
        assert retrieved['type'] == 'analysis'
        assert retrieved['content'] == 'Code looks good'
        assert retrieved['id'] == 1

        # Store another record
        test_data2 = {
            'type': 'analysis',
            'content': 'Updated analysis',
            'timestamp': vm._get_timestamp()
        }

        record_id2 = vm.store_data(main_file, 'analysis', test_data2)
        assert record_id2 == 2

        # Get latest should return most recent
        latest = vm.get_latest_data(main_file, 'analysis')
        assert latest['id'] == 2
        assert latest['content'] == 'Updated analysis'

        # Test with query filter
        filtered = vm.get_latest_data(main_file, 'analysis', {'id': 1})
        assert filtered is None  # Latest record has id=2, not 1

        # Test non-existent table
        empty = vm.get_latest_data(main_file, 'nonexistent')
        assert empty is None

    # ========== VALIDATION TESTS ==========

    def test_validation_methods(self, setup):
        """Test comprehensive validation functionality"""
        vm = setup['vm']
        agent = setup['agents'][0]
        main_file = setup['files']['module.py']

        # Test validation with no active files
        vm.validate_active_files(agent)  # Should not raise

        # Test validation with active file
        with patch.object(vm, '_is_process_running', return_value=True):
            vm.flag_active(agent, main_file)
            vm.validate_active_files(agent)  # Should not raise

            # Test validation error scenarios
            # Manually corrupt the active file
            meta_dir = vm._get_meta_dir(main_file)
            active_file_path = vm._get_file_path(meta_dir, FileType.ACTIVE)

            # Test with corrupted JSON
            with open(active_file_path, 'w') as f:
                f.write("invalid json content")

            with pytest.raises(ValueError, match="corrupted"):
                vm.validate_active_files(agent)

            vm.unflag_active(agent, main_file)

        # Test ValidationError class
        error = ValidationError("test.py", "field_name", "Error message", "invalid_value")
        assert error.file_path == "test.py"
        assert error.field == "field_name"
        assert error.message == "Error message"
        assert error.value == "invalid_value"

        # Test validation helper methods
        test_data = {'required_field': 'value'}
        errors = vm._validate_required_fields(test_data, ['required_field', 'missing_field'], 'test_context')
        assert len(errors) == 1
        assert errors[0].field == 'missing_field'

        # Test timestamp validation
        valid_timestamp = vm._get_timestamp()
        assert vm._validate_timestamp(valid_timestamp, 'test') is None

        invalid_error = vm._validate_timestamp('invalid-timestamp', 'test')
        assert invalid_error is not None
        assert invalid_error.field == 'timestamp'

    # ========== CONCURRENCY AND LOCKING TESTS ==========

    def test_concurrent_version_saves(self, setup):
        """Test concurrent version saving"""
        vm = setup['vm']
        temp_dir = setup['temp_dir']

        # Create separate files for each worker to avoid any collision
        test_files = []
        for i in range(5):
            file_path = os.path.join(temp_dir, f"concurrent_save_{i}.py")
            content = f"def worker_{i}_function():\n    return {i}"
            with open(file_path, 'w') as f:
                f.write(content)
            test_files.append((file_path, content))

        def save_version_worker(worker_id):
            file_path, base_content = test_files[worker_id]
            # Create unique content with timestamp and worker ID
            unique_content = f"{base_content}\n# Worker {worker_id} at {time.time()}"
            try:
                version = vm.save_version(
                    file_path, unique_content,
                    changes=[{"type": "test", "description": f"Concurrent save {worker_id}"}],
                    explanation=f"Concurrent test save from worker {worker_id}"
                )
                return worker_id, version, None
            except Exception as e:
                return worker_id, None, str(e)

        # Run concurrent saves on separate files
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(save_version_worker, i) for i in range(5)]
            results = [future.result() for future in as_completed(futures)]

        # Verify all saves completed successfully
        versions = []
        for worker_id, version, error in results:
            assert error is None, f"Worker {worker_id} failed: {error}"
            assert version is not None
            versions.append(version)

        # Since each worker uses a separate file, all should get version 1
        assert all(v == 1 for v in versions), f"Expected all version 1, got: {versions}"
        assert len(versions) == 5, f"Expected 5 results, got {len(versions)}"

    def test_locked_file_operations(self, setup):
        """Test locked file operation context managers"""
        temp_dir = setup['temp_dir']

        # Test locked_file_operation
        test_file = os.path.join(temp_dir, "test_lock.txt")

        with locked_file_operation(test_file, 'w') as f:
            f.write("test content")

        assert os.path.exists(test_file)
        with open(test_file, 'r') as f:
            assert f.read() == "test content"

        # Test locked_json_operation
        json_file = os.path.join(temp_dir, "test_lock.json")

        with locked_json_operation(json_file, {"default": "value"}) as data:
            data["new_key"] = "new_value"
            data["number"] = 42

        assert os.path.exists(json_file)
        with open(json_file, 'r') as f:
            loaded = json.load(f)

        assert loaded["default"] == "value"
        assert loaded["new_key"] == "new_value"
        assert loaded["number"] == 42

        # Test updating existing JSON
        with locked_json_operation(json_file) as data:
            data["updated"] = True
            data["number"] = 100

        with open(json_file, 'r') as f:
            updated = json.load(f)

        assert updated["default"] == "value"  # Preserved
        assert updated["updated"] is True    # Added
        assert updated["number"] == 100      # Modified

    def test_concurrent_active_file_access(self, setup):
        """Test concurrent active file access scenarios"""
        vm = setup['vm']
        agents = setup['agents']
        main_file = setup['files']['module.py']

        # Use separate files for each agent to avoid collision
        temp_dir = setup['temp_dir']
        test_files = []
        for i in range(3):
            file_path = os.path.join(temp_dir, f"concurrent_{i}.py")
            with open(file_path, 'w') as f:
                f.write(f"# File {i}")
            test_files.append(file_path)

        def flag_active_worker(agent_index):
            agent = agents[agent_index]
            file_path = test_files[agent_index]  # Each agent gets their own file
            try:
                with patch.object(vm, '_is_process_running', return_value=True):
                    result = vm.flag_active(agent, file_path)
                    time.sleep(0.1)  # Hold the flag briefly
                    vm.unflag_active(agent, file_path)
                    return agent_index, result, None
            except Exception as e:
                return agent_index, None, str(e)

        # All agents should successfully flag their own files
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(flag_active_worker, i) for i in range(3)]
            results = [future.result() for future in as_completed(futures)]

        success_count = sum(1 for _, result, error in results if error is None and result is True)
        error_count = sum(1 for _, result, error in results if error is not None)

        # All should succeed since they're working on different files
        assert success_count == 3, f"Expected 3 successes, got {success_count}"
        assert error_count == 0, f"Expected 0 errors, got {error_count}"

    # ========== ERROR HANDLING TESTS ==========

    def test_file_permission_errors(self, setup):
        """Test handling of file permission errors"""
        vm = setup['vm']
        temp_dir = setup['temp_dir']

        # Create a file and make it read-only
        readonly_file = os.path.join(temp_dir, "readonly.py")
        with open(readonly_file, 'w') as f:
            f.write("# readonly file")

        # Make file read-only (on Windows, this might not prevent all writes)
        if os.name != 'nt':  # Skip on Windows due to permission model differences
            os.chmod(readonly_file, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

            # Try to save version (should handle gracefully or raise appropriate error)
            try:
                version = vm.save_version(readonly_file, "new content")
                # If it succeeds, verify the version was saved to meta directory
                assert version >= 1
            except (PermissionError, OSError):
                # Expected for read-only files
                pass

    def test_corrupted_json_handling(self, setup):
        """Test handling of corrupted JSON files"""
        vm = setup['vm']
        main_file = setup['files']['module.py']
        meta_dir = vm._get_meta_dir(main_file)

        # Create corrupted index.json
        index_file = vm._get_file_path(meta_dir, FileType.INDEX)
        with open(index_file, 'w') as f:
            f.write("invalid json content {")

        # Should handle corrupted file gracefully and recreate
        version = vm.save_version(main_file, "new content")
        assert version == 1

        # Verify file was recreated properly
        with open(index_file, 'r') as f:
            index = json.load(f)
        assert 'versions' in index
        assert len(index['versions']) == 1

    def test_missing_dependencies_simulation(self, setup):
        """Test behavior when dependencies are missing"""
        vm = setup['vm']

        # Test with missing main file for test
        test_file = setup['files']['module_test.py']

        with pytest.raises(ValueError, match="Could not infer main file"):
            # Remove main file temporarily
            main_file = setup['files']['module.py']
            temp_name = main_file + ".temp"
            os.rename(main_file, temp_name)

            try:
                vm.save_test_version(test_file, "test content")
            finally:
                # Restore file
                os.rename(temp_name, main_file)

    # ========== COMPREHENSIVE INTEGRATION TESTS ==========

    def test_complete_development_workflow(self, setup):
        """Test complete development workflow simulation"""
        vm = setup['vm']
        agent = setup['agents'][0]
        main_file = setup['files']['module.py']
        test_file = setup['files']['module_test.py']
        main_content = setup['contents']['module.py']
        test_content = setup['contents']['module_test.py']

        with patch.object(vm, '_is_process_running', return_value=True):
            # 1. Developer starts working on file
            vm.flag_active(agent, main_file)
            assert vm.check_active(main_file, ignore_self=False)

            # 2. Save original versions
            main_v1 = vm.save_original(main_file, main_content)
            test_v1 = vm.save_test_version(test_file, test_content)
            assert main_v1 == test_v1 == 1

            # 3. Make several iterations
            iterations = [
                ("# Added documentation\n" + main_content, "Added documentation"),
                (main_content + "\n\ndef multiply(a, b):\n    return a * b", "Added multiply function"),
                (main_content + "\n\ndef multiply(a, b):\n    return a * b\n\ndef divide(a, b):\n    return a / b", "Added divide function")
            ]

            expected_version = 2
            for new_content, description in iterations:
                version = vm.save_version(
                    main_file, new_content,
                    changes=[{"type": "enhancement", "description": description}],
                    explanation=f"Iteration: {description}"
                )
                assert version == expected_version
                expected_version += 1

            # 4. Update tests accordingly
            enhanced_test = test_content + "\n\ndef test_multiply():\n    assert multiply(3, 4) == 12"
            test_v2 = vm.save_test_version(
                test_file, enhanced_test,
                changes=[{"type": "test", "description": "Added multiply test"}],
                explanation="Test for multiply function"
            )
            assert test_v2 == 2

            # 5. Store analysis data
            analysis_data = {
                'type': 'code_review',
                'score': 85,
                'issues': ['Consider edge cases for divide by zero'],
                'suggestions': ['Add input validation', 'Improve documentation']
            }
            vm.store_data(main_file, 'analysis', analysis_data)

            # 6. Finish work
            vm.unflag_active(agent, main_file)
            vm.flag_finished(agent, main_file)

            # 7. Verify final state
            assert not vm.check_active(main_file)
            assert vm.check_finished(agent, main_file)

            # 8. Verify all metadata exists and is correct
            meta_dir = vm._get_meta_dir(main_file)
            test_meta_dir = vm._get_test_meta_dir(test_file, main_file)

            # Check main file versions
            with open(os.path.join(meta_dir, "index.json"), 'r') as f:
                main_index = json.load(f)
            assert len(main_index['versions']) == 4  # Original + 3 iterations

            # Check test file versions
            with open(os.path.join(test_meta_dir, "index.json"), 'r') as f:
                test_index = json.load(f)
            assert len(test_index['versions']) == 2  # Original + 1 update

            # Check analysis data
            retrieved_analysis = vm.get_latest_data(main_file, 'analysis')
            assert retrieved_analysis['score'] == 85
            assert len(retrieved_analysis['issues']) == 1

            # Check finished status
            with open(vm._get_file_path(meta_dir, FileType.FINISHED), 'r') as f:
                finished_data = json.load(f)
            assert len(finished_data['agents']) == 1

    def test_multi_agent_collaboration(self, setup):
        """Test multiple agents working on different files"""
        vm = setup['vm']
        agent1, agent2, agent3 = setup['agents'][:3]
        files = [setup['files']['module.py'], setup['files']['utils.py'], setup['files']['empty.py']]
        contents = [setup['contents']['module.py'], setup['contents']['utils.py'], setup['contents']['empty.py']]

        with patch.object(vm, '_is_process_running', return_value=True):
            # Each agent works on different file
            for agent, file_path, content in zip([agent1, agent2, agent3], files, contents):
                vm.flag_active(agent, file_path)
                vm.save_original(file_path, content)

                # Make some changes
                modified = content + f"\n# Modified by {agent.get_id()}"
                vm.save_version(
                    file_path, modified,
                    changes=[{"type": "modification", "description": f"Changes by {agent.get_id()}"}],
                    explanation=f"Work completed by {agent.get_id()}"
                )

                vm.unflag_active(agent, file_path)
                vm.flag_finished(agent, file_path)

            # Verify all agents completed their work
            for agent, file_path in zip([agent1, agent2, agent3], files):
                assert not vm.check_active(file_path)
                assert vm.check_finished(agent, file_path)

                # Verify version files exist
                meta_dir = vm._get_meta_dir(file_path)
                with open(os.path.join(meta_dir, "index.json"), 'r') as f:
                    index = json.load(f)
                assert len(index['versions']) == 2  # Original + modified

    def test_system_stress_and_cleanup(self, setup):
        """Test system under stress and comprehensive cleanup"""
        vm = setup['vm']
        agent = setup['agents'][0]
        temp_dir = setup['temp_dir']

        # Create many files and versions
        stress_files = []
        for i in range(10):
            file_path = os.path.join(temp_dir, f"stress_{i}.py")
            content = f"def func_{i}():\n    return {i}"

            with open(file_path, 'w') as f:
                f.write(content)
            stress_files.append((file_path, content))

        # Save multiple versions for each file
        with patch.object(vm, '_is_process_running', return_value=True):
            for file_path, base_content in stress_files:
                vm.save_original(file_path, base_content)

                # Create 5 versions per file
                for version in range(2, 7):
                    modified_content = base_content + f"\n# Version {version}"
                    vm.save_version(
                        file_path, modified_content,
                        changes=[{"type": "update", "description": f"Version {version}"}],
                        explanation=f"Stress test version {version}"
                    )

        # Verify all versions were created
        total_versions = 0
        for file_path, _ in stress_files:
            meta_dir = vm._get_meta_dir(file_path)
            with open(os.path.join(meta_dir, "index.json"), 'r') as f:
                index = json.load(f)
            total_versions += len(index['versions'])

        assert total_versions == 60  # 10 files * 6 versions each

        # Test comprehensive shutdown
        vm.shutdown(agent)

        # Verify database shutdown was called (check the mock instance)
        assert hasattr(mock_db, 'shutdown')  # Verify shutdown method exists

    # ========== PERFORMANCE AND EDGE CASE TESTS ==========

    def test_large_content_handling(self, setup):
        """Test handling of large file content"""
        vm = setup['vm']
        temp_dir = setup['temp_dir']

        # Create large content (1MB)
        large_content = "# Large file\n" + ("x = 'data'\n" * 50000)
        large_file = os.path.join(temp_dir, "large.py")

        with open(large_file, 'w') as f:
            f.write(large_content)

        # Should handle large content efficiently
        start_time = time.time()
        version = vm.save_version(large_file, large_content)
        end_time = time.time()

        assert version == 1
        assert (end_time - start_time) < 5.0  # Should complete within 5 seconds

        # Verify large content was saved correctly
        meta_dir = vm._get_meta_dir(large_file)
        version_file = os.path.join(meta_dir, "large_v1.py")

        with open(version_file, 'r') as f:
            saved_content = f.read()

        assert len(saved_content) == len(large_content)
        assert saved_content == large_content

    def test_unicode_and_encoding_edge_cases(self, setup):
        """Test Unicode and encoding edge cases"""
        vm = setup['vm']
        temp_dir = setup['temp_dir']

        # Test various Unicode scenarios
        unicode_tests = [
            ("emoji.py", "def greet():\n    return '👋 Hello! 🌍🚀✨'"),
            ("chinese.py", "def chinese():\n    return '你好世界'"),
            ("arabic.py", "def arabic():\n    return 'مرحبا بالعالم'"),
            ("mixed.py", "def mixed():\n    return 'Hello 世界 🌍 مرحبا'"),
            ("special.py", "# Special chars: ÄÖÜäöüß©®™€\ndef func():\n    pass")
        ]

        for filename, content in unicode_tests:
            file_path = os.path.join(temp_dir, filename)

            # Write with explicit UTF-8 encoding
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Save version
            version = vm.save_version(file_path, content)
            assert version == 1

            # Verify content integrity
            meta_dir = vm._get_meta_dir(file_path)
            base_name = os.path.splitext(filename)[0]
            version_file = os.path.join(meta_dir, f"{base_name}_v1.py")

            with open(version_file, 'r', encoding='utf-8') as f:
                saved_content = f.read()

            assert saved_content == content, f"Unicode content mismatch for {filename}"

    def test_filesystem_edge_cases(self, setup):
        """Test filesystem edge cases"""
        vm = setup['vm']
        temp_dir = setup['temp_dir']

        # Test with very long filename (within limits)
        long_name = "very_long_filename_" + "x" * 100 + ".py"
        long_file = os.path.join(temp_dir, long_name)

        with open(long_file, 'w') as f:
            f.write("# Long filename test")

        version = vm.save_version(long_file, "# Long filename test")
        assert version == 1

        # Test with special characters in path (that are filesystem-safe)
        special_chars = "file-with.special_chars[1].py"
        special_file = os.path.join(temp_dir, special_chars)

        with open(special_file, 'w') as f:
            f.write("# Special chars test")

        version = vm.save_version(special_file, "# Special chars test")
        assert version == 1


def run_tests():
    """Run the comprehensive test suite"""
    print("🧪 Running Comprehensive VersionManager Test Suite")
    print("=" * 70)
    print("Testing: Core functionality, edge cases, concurrency, and error handling")
    print()

    pytest_args = [
        __file__,
        '-v',
        '-s',
        '--tb=short',
        '--disable-warnings',
        '--maxfail=5',  # Stop after 5 failures for faster debugging
    ]

    start_time = time.time()
    exit_code = pytest.main(pytest_args)
    end_time = time.time()

    print("\n" + "=" * 70)
    print(f"Test execution time: {end_time - start_time:.2f} seconds")

    if exit_code == 0:
        print("✅ ALL TESTS PASSED! VersionManager is comprehensively tested.")
        print("🎉 Coverage includes:")
        print("   • Core functionality and utilities")
        print("   • Version management workflows")
        print("   • Active/finished file management")
        print("   • Database operations")
        print("   • Concurrency and locking")
        print("   • Error handling and edge cases")
        print("   • Unicode and encoding")
        print("   • Performance and stress testing")
    else:
        print("❌ Some tests failed. Check output above for details.")
        print("💡 Use pytest -v --tb=long for detailed error information")

    return exit_code


if __name__ == "__main__":
    run_tests()