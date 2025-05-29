#!/usr/bin/env python3
"""
Efficient VersionManager tests with minimal mocks and maximum real object usage.
Run with: python test_version_manager.py
"""

import pytest
import os
import json
import tempfile
import shutil
import sys
import time
import threading
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_minimal_mocks():
    """Setup only essential mocks, use real objects wherever possible"""
    # Mock only external dependencies that can't be tested
    mock_psutil = Mock()
    mock_psutil.NoSuchProcess = Exception
    mock_psutil.AccessDenied = Exception
    mock_psutil.Process.side_effect = lambda pid: Mock(
        username=lambda: "testuser",
        name=lambda: "python",
        cmdline=lambda: ["python", "test.py"]
    )

    # Mock portalocker to avoid file locking issues in tests
    mock_portalocker = Mock()
    mock_portalocker.LOCK_EX = 2
    mock_portalocker.LOCK_SH = 1
    mock_portalocker.LOCK_NB = 4
    mock_portalocker.LockException = Exception
    # Make lock operations no-ops for testing
    mock_portalocker.lock = Mock()
    mock_portalocker.unlock = Mock()

    # Simple config constants and function
    config = Mock()
    config.META_DIR_EXT = "meta"
    config.TEST_FILE_EXT = "_test.py"

    # Simple logger that prints to console
    logger = Mock()
    logger.debug = logger.info = logger.warning = logger.error = lambda msg: None

    # Simple hash function
    parsing = Mock()
    parsing.hash_string = lambda x: f"hash_{abs(hash(x)):016x}"

    # Mock only the imports, let everything else be real
    sys.modules.update({
        'psutil': mock_psutil,
        'portalocker': mock_portalocker,
        'rome.config': config,
        'rome.logger': Mock(get_logger=lambda: logger),
        'rome.parsing': parsing
    })

    return mock_psutil, logger

# Setup and import
mock_psutil, logger = setup_minimal_mocks()

# Mock set_attributes_from_config in the config module
def mock_set_attributes_from_config(obj, attrs):
    """Mock implementation that sets default values"""
    defaults = {'lock_timeout': 5.0, 'max_retries': 3, 'retry_delay': 0.1}
    for attr in attrs:
        if attr in defaults:
            setattr(obj, attr, defaults[attr])

# Add the function to the mocked config module
sys.modules['rome.config'].set_attributes_from_config = mock_set_attributes_from_config

# Import after mocking - this will use real DatabaseManager and real file operations
from rome.version_manager import VersionManager, ValidationError, FileType
from rome.database import locked_file_operation, locked_json_operation

class TestVersionManager:
    """Efficient test suite using real objects where possible"""

    @pytest.fixture
    def vm_setup(self):
        """Compact setup with real filesystem and database"""
        temp_dir = tempfile.mkdtemp()
        vm = VersionManager()

        # Create test files
        files = {}
        test_data = [
            ("main.py", "def add(a, b):\n    return a + b"),
            ("utils.py", "def multiply(x, y):\n    return x * y"),
            ("main_test.py", "def test_add():\n    assert add(2, 3) == 5")
        ]

        for name, content in test_data:
            path = os.path.join(temp_dir, name)
            with open(path, 'w') as f:
                f.write(content)
            files[name] = path

        # Single agent for most tests
        agent = Mock()
        agent.get_id.return_value = f"test_agent_{os.getpid()}_{os.getpid()}"  # Ensure 3+ parts

        yield temp_dir, vm, files, agent, {name: content for name, content in test_data}

        # Cleanup
        try:
            vm.shutdown(agent)
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass

        # Restore original methods if they were patched
        try:
            if hasattr(TimeoutLockedJSONStorage, '_original_init'):
                TimeoutLockedJSONStorage.__init__ = TimeoutLockedJSONStorage._original_init
                TimeoutLockedJSONStorage.read = TimeoutLockedJSONStorage._original_read
                TimeoutLockedJSONStorage.write = TimeoutLockedJSONStorage._original_write
        except:
            pass

    def test_core_functionality(self, vm_setup):
        """Test all core functionality in one efficient test"""
        temp_dir, vm, files, agent, contents = vm_setup
        main_file = files["main.py"]
        main_content = contents["main.py"]

        # Test meta directory creation
        meta_dir = vm._get_meta_dir(main_file)
        assert meta_dir.endswith(".meta")
        assert os.path.exists(meta_dir)

        # Test version saving workflow
        v1 = vm.save_original(main_file, main_content)
        assert v1 == 1

        # Save same content (should return existing)
        v2 = vm.save_version(main_file, main_content)
        assert v2 == 1

        # Save modified content
        modified = main_content + "\ndef subtract(a, b):\n    return a - b"
        v3 = vm.save_version(main_file, modified,
                           changes=[{"type": "add", "desc": "subtract func"}],
                           explanation="Added subtraction")
        assert v3 == 2

        # Verify files exist
        index_file = os.path.join(meta_dir, "index.json")
        version_file = os.path.join(meta_dir, "main_v2.py")
        assert os.path.exists(index_file)
        assert os.path.exists(version_file)

        # Verify content
        with open(version_file) as f:
            assert f.read() == modified

        with open(index_file) as f:
            index = json.load(f)
        assert len(index["versions"]) == 2
        assert index["versions"][1]["version"] == 2

    def test_active_file_management(self, vm_setup):
        """Test active file lifecycle"""
        temp_dir, vm, files, agent, contents = vm_setup
        main_file = files["main.py"]

        with patch.object(vm, '_is_process_running', return_value=True):
            # Initial state
            assert not vm.check_active(main_file)
            assert not vm._has_active_files()

            # Flag active
            assert vm.flag_active(agent, main_file)
            assert vm.check_active(main_file, ignore_self=False)
            assert not vm.check_active(main_file, ignore_self=True)
            assert vm._has_active_files()

            # Cannot flag another file
            other_file = files["utils.py"]
            with pytest.raises(RuntimeError, match="already has active"):
                vm.flag_active(agent, other_file)

            # Unflag
            assert vm.unflag_active(agent, main_file)
            assert not vm.check_active(main_file)
            assert not vm._has_active_files()

    def test_test_file_workflow(self, vm_setup):
        """Test test file versioning"""
        temp_dir, vm, files, agent, contents = vm_setup
        test_file = files["main_test.py"]
        main_file = files["main.py"]
        test_content = contents["main_test.py"]

        # Save test version with inference
        v1 = vm.save_test_version(test_file, test_content)
        assert v1 == 1

        # Save with explicit main file
        modified = test_content + "\ndef test_subtract():\n    assert subtract(5, 3) == 2"
        v2 = vm.save_test_version(test_file, modified, main_file_path=main_file)
        assert v2 == 2

        # Verify test meta structure
        test_meta_dir = vm._get_test_meta_dir(test_file, main_file)
        assert os.path.exists(test_meta_dir)
        assert os.path.exists(os.path.join(test_meta_dir, "index.json"))

    def test_finished_flag_workflow(self, vm_setup):
        """Test finished file flagging"""
        temp_dir, vm, files, agent, contents = vm_setup
        main_file = files["main.py"]

        # Initially not finished
        assert not vm.check_finished(agent, main_file)

        # Flag finished
        vm.flag_finished(agent, main_file)
        assert vm.check_finished(agent, main_file)

        # Multiple agents
        agent2 = Mock()
        agent2.get_id.return_value = "agent2_host_123"
        vm.flag_finished(agent2, main_file)
        assert vm.check_finished(agent2, main_file)

        # Verify file structure
        meta_dir = vm._get_meta_dir(main_file)
        finished_file = os.path.join(meta_dir, "finished.json")
        with open(finished_file) as f:
            data = json.load(f)
        assert len(data["agents"]) == 2

    def test_database_operations(self, vm_setup):
        """Test database interface"""
        temp_dir, vm, files, agent, contents = vm_setup
        main_file = files["main.py"]

        # Store data
        test_data = {"type": "analysis", "score": 95}
        doc_id = vm.store_data(main_file, "analysis", test_data)
        assert doc_id > 0

        # Retrieve data
        retrieved = vm.get_data(main_file, "analysis")
        assert retrieved["type"] == "analysis"
        assert retrieved["score"] == 95

        # Store overwrites (clear_table behavior)
        new_data = {"type": "review", "score": 88}
        vm.store_data(main_file, "analysis", new_data)
        latest = vm.get_data(main_file, "analysis")
        assert latest["type"] == "review"
        assert latest["score"] == 88

    def test_validation_and_edge_cases(self, vm_setup):
        """Test validation and edge cases"""
        temp_dir, vm, files, agent, contents = vm_setup
        main_file = files["main.py"]

        # Validation with no active files
        vm.validate_active_files(agent)

        # PID extraction edge cases - agent ID needs >= 3 parts separated by _
        assert vm._get_pid_from_agent_id("test_agent_host_123") == 123
        assert vm._get_pid_from_agent_id("agent_123") is None  # Only 2 parts
        assert vm._get_pid_from_agent_id("invalid") is None
        assert vm._get_pid_from_agent_id("") is None

        # File path cleaning
        assert vm._clean_file_path("file.py.meta") == "file.py"
        assert vm._clean_file_path("file.py") == "file.py"

        # Version utilities
        index = {"versions": [{"version": 1, "hash": "abc"}, {"version": 3, "hash": "def"}]}
        assert vm._find_existing_version_by_hash(index, "abc") == 1
        assert vm._find_existing_version_by_hash(index, "xyz") is None
        assert vm._get_next_version_number(index) == 4
        assert vm._get_next_version_number({}) == 1

    def test_stale_process_cleanup(self, vm_setup):
        """Test cleanup of stale processes"""
        temp_dir, vm, files, agent, contents = vm_setup
        main_file = files["main.py"]

        # Create stale active file
        meta_dir = vm._get_meta_dir(main_file)
        active_file = os.path.join(meta_dir, "active.json")
        stale_data = {
            "agent_id": "stale_agent_host_99999",  # Non-existent PID with proper format
            "timestamp": vm._get_timestamp(),
            "file_path": main_file
        }
        with open(active_file, 'w') as f:
            json.dump(stale_data, f)

        # Should clean up automatically
        assert not vm.check_active(main_file)
        assert not os.path.exists(active_file)

    def test_concurrent_operations(self, vm_setup):
        """Test thread safety with real concurrency"""
        temp_dir, vm, files, agent, contents = vm_setup

        # Create separate files for concurrent access
        concurrent_files = []
        for i in range(3):
            path = os.path.join(temp_dir, f"concurrent_{i}.py")
            with open(path, 'w') as f:
                f.write(f"# File {i}")
            concurrent_files.append(path)

        def save_versions(file_index):
            file_path = concurrent_files[file_index]
            results = []
            for v in range(3):
                content = f"# File {file_index} version {v}\ndef func_{v}(): pass"
                try:
                    version = vm.save_version(file_path, content)
                    results.append(version)
                except Exception as e:
                    results.append(f"Error: {e}")
            return file_index, results

        # Run concurrent saves
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(save_versions, i) for i in range(3)]
            results = [f.result() for f in as_completed(futures)]

        # Verify all succeeded
        for file_index, versions in results:
            assert all(isinstance(v, int) for v in versions), f"File {file_index} had errors: {versions}"
            assert len(versions) == 3

    def test_unicode_and_special_content(self, vm_setup):
        """Test special content handling"""
        temp_dir, vm, files, agent, contents = vm_setup

        special_cases = [
            ("", "empty.py"),
            ("def greet():\n    return '你好🌍'", "unicode.py"),
            ("x = 'content'\n" * 1000, "large.py"),
            ('"""Multi\nline\nstring"""', "multiline.py")
        ]

        for content, filename in special_cases:
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            version = vm.save_version(file_path, content)
            assert version == 1

            # Verify saved correctly
            meta_dir = vm._get_meta_dir(file_path)
            base = os.path.splitext(filename)[0]
            version_file = os.path.join(meta_dir, f"{base}_v1.py")
            with open(version_file, 'r', encoding='utf-8') as f:
                saved = f.read()
            assert saved == content

    def test_error_handling(self, vm_setup):
        """Test error conditions"""
        temp_dir, vm, files, agent, contents = vm_setup
        test_file = files["main_test.py"]

        # Test file without main file
        nonexistent_test = os.path.join(temp_dir, "orphan_test.py")
        with open(nonexistent_test, 'w') as f:
            f.write("# orphaned test")

        with pytest.raises(ValueError, match="Could not infer main file"):
            vm.save_test_version(nonexistent_test, "test content")

        # Test corrupted JSON handling
        main_file = files["main.py"]
        meta_dir = vm._get_meta_dir(main_file)
        index_file = os.path.join(meta_dir, "index.json")

        # Create corrupted file
        with open(index_file, 'w') as f:
            f.write("invalid json {")

        # Should handle gracefully
        version = vm.save_version(main_file, "new content")
        assert version == 1

    def test_complete_workflow_integration(self, vm_setup):
        """Test complete development workflow"""
        temp_dir, vm, files, agent, contents = vm_setup
        main_file = files["main.py"]
        test_file = files["main_test.py"]

        with patch.object(vm, '_is_process_running', return_value=True):
            # Start working
            vm.flag_active(agent, main_file)

            # Save original
            vm.save_original(main_file, contents["main.py"])
            vm.save_test_version(test_file, contents["main_test.py"])

            # Make changes
            for i, change in enumerate(["# Comment", "def new_func(): pass"], 1):
                new_content = contents["main.py"] + f"\n{change}"
                vm.save_version(main_file, new_content,
                              changes=[{"type": "update", "desc": f"Change {i}"}])

            # Store analysis
            vm.store_data(main_file, "analysis", {"score": 90, "issues": []})

            # Finish
            vm.unflag_active(agent, main_file)
            vm.flag_finished(agent, main_file)

            # Verify final state
            assert not vm.check_active(main_file)
            assert vm.check_finished(agent, main_file)

            analysis = vm.get_data(main_file, "analysis")
            assert analysis["score"] == 90


def run_tests():
    """Simple test runner with dependency checking"""
    try:
        import tinydb, portalocker
        print("✓ Dependencies available")
    except ImportError as e:
        print(f"✗ Missing: {e}\nInstall: pip install tinydb portalocker")
        return False

    # Run with pytest if available, otherwise simple runner
    try:
        import pytest
        exit_code = pytest.main([__file__, '-v', '--tb=short'])
        return exit_code == 0
    except ImportError:
        print("Running without pytest...")
        # Simple test runner
        test_instance = TestVersionManager()
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]

        passed = failed = 0
        for method_name in sorted(test_methods):
            try:
                print(f"Running {method_name}...", end=' ')
                # Setup fixture manually
                setup_gen = test_instance.vm_setup()
                setup_data = next(setup_gen)

                # Run test
                getattr(test_instance, method_name)(setup_data)

                # Cleanup
                try:
                    next(setup_gen)
                except StopIteration:
                    pass

                print("✓")
                passed += 1
            except Exception as e:
                print(f"✗ {e}")
                failed += 1

        print(f"\nResults: {passed} passed, {failed} failed")
        return failed == 0


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
