#!/usr/bin/env python3
"""
Comprehensive pytest suite for VersionManager
Run with: python test_version_manager.py
"""

import pytest
import os
import json
import tempfile
import shutil
import sys
from unittest.mock import Mock, patch, MagicMock
import datetime
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dependencies
class MockConfig:
    META_DIR_EXT = "meta"

class MockLogger:
    def debug(self, msg): print(f"DEBUG: {msg}")
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")

def get_logger():
    return MockLogger()

# Create mock modules before importing VersionManager
mock_psutil = MagicMock()
mock_portalocker = MagicMock()
mock_portalocker.LOCK_EX = 2
mock_portalocker.lock = MagicMock()

# Mock the modules that VersionManager depends on
sys.modules['psutil'] = mock_psutil
sys.modules['portalocker'] = mock_portalocker

# Mock the rome modules
mock_config = MagicMock()
mock_config.META_DIR_EXT = "meta"
sys.modules['rome.config'] = mock_config
sys.modules['rome.logger'] = MagicMock()
sys.modules['rome.logger'].get_logger = get_logger
sys.modules['rome.parsing'] = MagicMock()
sys.modules['rome.parsing'].hash_string = lambda x: f"hash_{hash(x):064x}"[-64:]

# Now import the actual VersionManager
from rome.version_manager import VersionManager, ValidationError, FileType, locked_file_operation, locked_json_operation
print("✅ Successfully imported VersionManager")


class TestVersionManager:
    """Test suite for VersionManager class"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def vm(self):
        """Create a VersionManager instance"""
        return VersionManager()

    @pytest.fixture
    def sample_file(self, temp_dir):
        """Create a sample file for testing"""
        file_path = os.path.join(temp_dir, "sample.py")
        content = "def hello():\n    return 'world'"
        with open(file_path, 'w') as f:
            f.write(content)
        return file_path, content

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent with chat completion"""
        agent = Mock()
        # Use current process PID to ensure process checks work correctly
        agent.get_id.return_value = f"test_agent_{os.getpid()}"
        agent.chat_completion.return_value = "Test analysis result"
        agent.role = "Test role"
        return agent

    @pytest.fixture
    def test_file_pair(self, temp_dir):
        """Create a main file and corresponding test file"""
        main_file = os.path.join(temp_dir, "module.py")
        test_file = os.path.join(temp_dir, "module_test.py")

        main_content = "def add(a, b):\n    return a + b"
        test_content = "def test_add():\n    assert add(2, 3) == 5"

        with open(main_file, 'w') as f:
            f.write(main_content)

        return main_file, test_file, main_content, test_content

    # Utility Tests
    def test_init(self):
        """Test VersionManager initialization"""
        vm = VersionManager()
        assert vm.config == {}
        assert vm.logger is not None
        assert vm.active_files == set()

        config = {"test": "value"}
        vm = VersionManager(config)
        assert vm.config == config

    def test_timestamp_and_hash(self, vm):
        """Test timestamp and hash generation"""
        timestamp = vm._get_timestamp()
        assert isinstance(timestamp, str)
        datetime.datetime.fromisoformat(timestamp)

        content = "test content"
        hash1 = vm._get_content_hash(content)
        hash2 = vm._get_content_hash(content)
        hash3 = vm._get_content_hash("different content")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 64

    def test_meta_directories(self, vm, temp_dir):
        """Test meta directory creation and management"""
        file_path = os.path.join(temp_dir, "test.py")
        meta_dir = vm._get_meta_dir(file_path)

        expected = f"{file_path}.{mock_config.META_DIR_EXT}"
        assert meta_dir == expected
        assert os.path.exists(meta_dir)

    def test_file_path_utilities(self, vm, temp_dir):
        """Test file path utility methods"""
        # Test main file inference
        main_file = os.path.join(temp_dir, "module.py")
        test_file = os.path.join(temp_dir, "module_test.py")

        with open(main_file, 'w') as f:
            f.write("def main(): pass")

        inferred = vm._infer_main_file_from_test(test_file)
        assert inferred == main_file

        # Test non-test file
        regular_file = os.path.join(temp_dir, "regular.py")
        assert vm._infer_main_file_from_test(regular_file) is None

        # Test clean file path
        dirty_path = f"/test/file.py.{mock_config.META_DIR_EXT}"
        clean_path = vm._clean_file_path(dirty_path)
        assert clean_path == "/test/file.py"

    # Version Management Tests
    def test_version_utilities(self, vm):
        """Test version management utility methods"""
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

        # Test next version number
        assert vm._get_next_version_number({}) == 1
        assert vm._get_next_version_number({'versions': []}) == 1
        assert vm._get_next_version_number(index) == 4

    def test_version_metadata_creation(self, vm):
        """Test version metadata creation"""
        metadata = vm._create_version_metadata(
            file_path="/test/file.py",
            content_hash="abc123",
            version_number=1,
            changes=[{"type": "fix", "description": "bug fix"}],
            explanation="Test version",
            main_file_path="/test/main.py"
        )

        assert metadata['version'] == 1
        assert metadata['file_path'] == "/test/file.py"
        assert metadata['hash'] == "abc123"
        assert metadata['changes'] == [{"type": "fix", "description": "bug fix"}]
        assert metadata['explanation'] == "Test version"
        assert metadata['main_file_path'] == "/test/main.py"
        assert 'timestamp' in metadata

    def test_save_version_workflow(self, vm, sample_file):
        """Test complete version saving workflow"""
        file_path, content = sample_file

        # Save original
        version = vm.save_original(file_path, content)
        assert version == 1

        # Save same content (should return existing version)
        version2 = vm.save_version(file_path, content)
        assert version2 == 1

        # Save different content
        new_content = "def hello():\n    return 'universe'"
        version3 = vm.save_version(file_path, new_content,
                                 changes=[{"type": "update", "description": "Changed return value"}],
                                 explanation="Updated greeting")
        assert version3 == 2

        # Verify files exist
        meta_dir = vm._get_meta_dir(file_path)
        assert os.path.exists(os.path.join(meta_dir, "index.json"))
        assert os.path.exists(os.path.join(meta_dir, "sample_v1.py"))
        assert os.path.exists(os.path.join(meta_dir, "sample_v2.py"))

        # Verify index content
        with open(os.path.join(meta_dir, "index.json"), 'r') as f:
            index = json.load(f)
        assert len(index['versions']) == 2
        assert index['versions'][0]['version'] == 1
        assert index['versions'][1]['version'] == 2

    def test_save_test_version(self, vm, test_file_pair):
        """Test saving test file versions"""
        main_file, test_file, main_content, test_content = test_file_pair

        # Save test version with explicit main file
        version = vm.save_test_version(test_file, test_content, main_file_path=main_file)
        assert version == 1

        # Save test version with inferred main file
        version2 = vm.save_test_version(test_file, test_content + "\n# updated")
        assert version2 == 2

        # Verify test meta directory structure
        main_meta_dir = vm._get_meta_dir(main_file)
        test_meta_dir = vm._get_test_meta_dir(test_file, main_file)
        assert os.path.exists(test_meta_dir)
        assert test_meta_dir.startswith(main_meta_dir)
        assert "module_test.py.meta" in test_meta_dir

    # Analysis Tests
    def test_analysis_workflow(self, vm, sample_file, mock_agent):
        """Test analysis saving and loading"""
        file_path, _ = sample_file

        # Save analysis
        analysis_path = vm.save_analysis(
            file_path=file_path,
            analysis="Code looks good",
            test_path="/test/test_file.py",
            exit_code=0,
            output="All tests passed"
        )

        assert os.path.exists(analysis_path)

        # Load analysis data
        analysis_data = vm._load_analysis(file_path)
        assert analysis_data is not None
        assert analysis_data['analysis'] == "Code looks good"
        assert analysis_data['test_path'] == "/test/test_file.py"
        assert analysis_data['exit_code'] == 0
        assert analysis_data['output'] == "All tests passed"

        # Load formatted analysis
        formatted = vm.load_analysis(file_path)
        assert formatted is not None
        assert 'Code looks good' in formatted
        assert 'All tests passed' in formatted
        assert 'IMPORTANT' in formatted

    def test_create_analysis(self, vm, mock_agent):
        """Test analysis creation with LLM"""
        original_content = "def add(a, b): return a + b"
        test_content = "def test_add(): assert add(2, 3) == 5"
        output = "All tests passed"
        exit_code = 0

        result = vm.create_analysis(mock_agent, original_content, test_content, output, exit_code)
        assert result == "Test analysis result"
        mock_agent.chat_completion.assert_called_once()

    # Active File Management Tests
    def test_active_file_management(self, vm, sample_file, mock_agent):
        """Test active file flagging and checking"""
        file_path, _ = sample_file

        # Initially not active
        assert not vm.check_active(file_path)
        assert not vm._has_active_files()

        # Flag as active
        with patch.object(vm, '_is_process_running', return_value=True):
            result = vm.flag_active(mock_agent, file_path)
            assert result is True
            # Use ignore_self=False to check the current process
            assert vm.check_active(file_path, ignore_self=False)
            assert vm._has_active_files()
            assert os.path.abspath(file_path) in vm._get_active_files()

            # Try to flag another file (should fail)
            other_file = os.path.join(os.path.dirname(file_path), "other.py")
            with open(other_file, 'w') as f:
                f.write("# other file")

            with pytest.raises(RuntimeError, match="already has active file"):
                vm.flag_active(mock_agent, other_file)

            # Unflag
            result = vm.unflag_active(mock_agent, file_path)
            assert result is True
            assert not vm.check_active(file_path, ignore_self=False)
            assert not vm._has_active_files()

    def test_stale_process_cleanup(self, vm, sample_file, mock_agent):
        """Test cleanup of stale process files"""
        file_path, _ = sample_file

        # Test with current process (should work)
        with patch.object(vm, '_is_process_running', return_value=True):
            # Flag as active
            vm.flag_active(mock_agent, file_path)

            # With ignore_self=True (default), should return False because it ignores current process
            is_active_ignore_self = vm.check_active(file_path)  # default ignore_self=True
            assert not is_active_ignore_self

            # With ignore_self=False, should return True because the process is running
            is_active_include_self = vm.check_active(file_path, ignore_self=False)
            assert is_active_include_self

            # Clean up
            vm.unflag_active(mock_agent, file_path)

        # Test actual stale process cleanup with different agent
        stale_agent = Mock()
        stale_agent.get_id.return_value = "stale_agent_99999"  # Non-existent PID

        # Manually create stale active file
        meta_dir = vm._get_meta_dir(file_path)
        active_file_path = vm._get_file_path(meta_dir, FileType.ACTIVE)
        stale_data = {
            'agent_id': stale_agent.get_id(),
            'timestamp': vm._get_timestamp(),
            'file_path': file_path
        }
        with open(active_file_path, 'w') as f:
            json.dump(stale_data, f)

        # Check should clean up the stale file (no mocking, so process won't be found)
        is_active = vm.check_active(file_path)
        assert not is_active
        assert not os.path.exists(active_file_path)

    # Finished File Management Tests
    def test_finished_file_management(self, vm, sample_file, mock_agent):
        """Test finished file flagging"""
        file_path, _ = sample_file

        # Initially not finished
        assert not vm.check_finished(mock_agent, file_path)

        # Flag as finished
        vm.flag_finished(mock_agent, file_path)
        assert vm.check_finished(mock_agent, file_path)

        # Flag again (should not duplicate)
        vm.flag_finished(mock_agent, file_path)
        assert vm.check_finished(mock_agent, file_path)

        # Verify finished file content
        meta_dir = vm._get_meta_dir(file_path)
        finished_file_path = vm._get_file_path(meta_dir, FileType.FINISHED)
        with open(finished_file_path, 'r') as f:
            finished_data = json.load(f)

        assert len(finished_data['agents']) == 1
        assert finished_data['agents'][0]['agent_id'] == mock_agent.get_id()

    # Process Management Tests
    def test_process_utilities(self, vm):
        """Test process-related utility methods"""
        # Test PID extraction
        agent_id = "test_agent_12345"
        pid = vm._get_pid_from_agent_id(agent_id)
        assert pid == 12345

        # Test invalid agent ID formats
        assert vm._get_pid_from_agent_id("invalid_format") is None
        assert vm._get_pid_from_agent_id("test_agent") is None
        assert vm._get_pid_from_agent_id("") is None

        # Test process running check
        with patch('psutil.Process') as mock_process:
            mock_proc = mock_process.return_value
            mock_proc.username.return_value = "testuser"
            mock_proc.name.return_value = "python"
            mock_proc.cmdline.return_value = ["python", "script.py"]

            with patch('os.getlogin', return_value="testuser"):
                result = vm._is_process_running(12345)
                assert isinstance(result, bool)

    # Validation Tests
    def test_validation_methods(self, vm, sample_file, mock_agent):
        """Test validation functionality"""
        file_path, _ = sample_file

        # Test validation with no active files
        vm.validate_active_files(mock_agent)  # Should not raise

        # Test validation with active file
        with patch.object(vm, '_is_process_running', return_value=True):
            vm.flag_active(mock_agent, file_path)
            vm.validate_active_files(mock_agent)  # Should not raise

        # Test validation error creation
        error = ValidationError("test.py", "field", "message", "value")
        assert error.file_path == "test.py"
        assert error.field == "field"
        assert error.message == "message"
        assert error.value == "value"

    # Context Manager Tests
    def test_locked_operations(self, vm, temp_dir):
        """Test locked file and JSON operations"""
        # Test locked_file_operation (standalone function)
        test_file = os.path.join(temp_dir, "test.txt")
        with locked_file_operation(test_file, 'w') as f:
            f.write("test content")

        assert os.path.exists(test_file)
        with open(test_file, 'r') as f:
            assert f.read() == "test content"

        # Test locked_json_operation (standalone function)
        json_file = os.path.join(temp_dir, "test.json")
        with locked_json_operation(json_file, {"default": "value"}) as data:
            data["new_key"] = "new_value"

        assert os.path.exists(json_file)
        with open(json_file, 'r') as f:
            loaded = json.load(f)
        assert loaded["default"] == "value"
        assert loaded["new_key"] == "new_value"

    # Integration and Edge Case Tests
    def test_edge_cases(self, vm, temp_dir):
        """Test edge cases and error conditions"""
        file_path = os.path.join(temp_dir, "edge_case.py")

        # Test with empty content
        version = vm.save_version(file_path, "")
        assert version == 1

        # Test with large content
        large_content = "x" * 10000
        version = vm.save_version(file_path, large_content)
        assert version == 2

        # Test with unicode content
        unicode_content = "def hello():\n    return '你好世界'"
        version = vm.save_version(file_path, unicode_content)
        assert version == 3

    def test_shutdown(self, vm, sample_file, mock_agent):
        """Test shutdown cleanup"""
        file_path, _ = sample_file

        with patch.object(vm, '_is_process_running', return_value=True):
            # Create some active files
            vm.flag_active(mock_agent, file_path)
            assert vm._has_active_files()

            # Shutdown
            vm.shutdown(mock_agent)
            assert not vm._has_active_files()

    def test_complete_integration_workflow(self, vm, test_file_pair, mock_agent):
        """Test complete integration workflow"""
        main_file, test_file, main_content, test_content = test_file_pair

        try:
            with patch.object(vm, '_is_process_running', return_value=True):
                # 1. Save original version
                version = vm.save_original(main_file, main_content)
                assert version == 1

                # 2. Save test version
                test_version = vm.save_test_version(test_file, test_content)
                assert test_version == 1

                # 3. Save analysis
                analysis_path = vm.save_analysis(
                    main_file, "Code analysis complete",
                    test_path=test_file, exit_code=0, output="Tests passed"
                )
                assert os.path.exists(analysis_path)

                # 4. Flag as active
                vm.flag_active(mock_agent, main_file)
                assert vm.check_active(main_file, ignore_self=False)

                # 5. Save updated version
                updated_content = main_content + "\n# Updated"
                updated_version = vm.save_version(
                    main_file, updated_content,
                    changes=[{"type": "enhancement", "description": "Added comment"}],
                    explanation="Minor update"
                )
                assert updated_version == 2

                # 6. Unflag active
                vm.unflag_active(mock_agent, main_file)
                assert not vm.check_active(main_file, ignore_self=False)

                # 7. Flag as finished
                vm.flag_finished(mock_agent, main_file)
                assert vm.check_finished(mock_agent, main_file)

                # 8. Verify all metadata exists
                main_meta_dir = vm._get_meta_dir(main_file)
                test_meta_dir = vm._get_test_meta_dir(test_file, main_file)

                assert os.path.exists(os.path.join(main_meta_dir, "index.json"))
                assert os.path.exists(os.path.join(main_meta_dir, "code_analysis.json"))
                assert os.path.exists(os.path.join(main_meta_dir, "finished.json"))
                assert os.path.exists(os.path.join(test_meta_dir, "index.json"))

        except Exception as e:
            pytest.fail(f"Integration workflow failed: {e}")


def run_tests():
    """Run all tests with proper setup"""
    print("Testing VersionManager (updated implementation)")
    print("=" * 70)

    # Configure pytest arguments
    pytest_args = [
        __file__,
        '-v',  # verbose
        '-s',  # no capture (show prints)
        '--tb=short',  # shorter traceback format
        '--disable-warnings',  # cleaner output
    ]

    exit_code = pytest.main(pytest_args)

    print("\n" + "=" * 70)
    if exit_code == 0:
        print("✅ All tests passed!")
        print("🎉 Updated VersionManager implementation tested successfully!")
    else:
        print("❌ Some tests failed!")
        print("💡 Check the error messages above for details")

    return exit_code


if __name__ == "__main__":
    # Allow running with: python test_version_manager.py
    run_tests()