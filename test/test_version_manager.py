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
import hashlib
import sys
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager
import datetime
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the dependencies that might not be available
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

# Mock the rome.config and rome.logger modules
mock_config = MagicMock()
mock_config.META_DIR_EXT = "meta"
sys.modules['rome.config'] = mock_config
sys.modules['rome.logger'] = MagicMock()
sys.modules['rome.logger'].get_logger = get_logger

# Now import the actual VersionManager
from rome.version_manager import VersionManager
print("✅ Successfully imported VersionManager from rome.versioning")


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
        """Create a mock agent"""
        agent = Mock()
        agent.get_id.return_value = "test_agent_12345"
        return agent

    def test_init(self):
        """Test VersionManager initialization"""
        vm = VersionManager()
        assert vm.config == {}
        assert vm.logger is not None

        config = {"test": "value"}
        vm = VersionManager(config)
        assert vm.config == config

    def test_get_meta_dir(self, vm, temp_dir):
        """Test meta directory creation"""
        file_path = os.path.join(temp_dir, "test.py")
        meta_dir = vm._get_meta_dir(file_path)

        expected = f"{file_path}.{mock_config.META_DIR_EXT}"
        assert meta_dir == expected
        assert os.path.exists(meta_dir)

    def test_load_json_safely(self, vm, temp_dir):
        """Test safe JSON loading"""
        # Test non-existent file
        result = vm._load_json_safely("nonexistent.json")
        assert result == {}

        result = vm._load_json_safely("nonexistent.json", {"default": "value"})
        assert result == {"default": "value"}

        # Test valid JSON file
        json_file = os.path.join(temp_dir, "test.json")
        test_data = {"key": "value"}
        with open(json_file, 'w') as f:
            json.dump(test_data, f)

        result = vm._load_json_safely(json_file)
        assert result == test_data

        # Test corrupted JSON file
        with open(json_file, 'w') as f:
            f.write("invalid json{")

        result = vm._load_json_safely(json_file, {"error": "handled"})
        assert result == {"error": "handled"}

    def test_save_json(self, vm, temp_dir):
        """Test JSON saving"""
        json_file = os.path.join(temp_dir, "output.json")
        test_data = {"test": "data", "number": 42}

        vm._save_json(json_file, test_data)

        assert os.path.exists(json_file)
        with open(json_file, 'r') as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data

    def test_get_content_hash(self, vm):
        """Test content hash generation"""
        content = "test content"
        hash1 = vm._get_content_hash(content)
        hash2 = vm._get_content_hash(content)
        hash3 = vm._get_content_hash("different content")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 64  # SHA256 hex length

    def test_get_timestamp(self, vm):
        """Test timestamp generation"""
        timestamp = vm._get_timestamp()
        assert isinstance(timestamp, str)
        # Should be ISO format
        datetime.datetime.fromisoformat(timestamp)

    def test_find_existing_version_by_hash(self, vm):
        """Test finding existing version by hash"""
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

    def test_get_next_version_number(self, vm):
        """Test version number generation"""
        # Empty index
        assert vm._get_next_version_number({}) == 1
        assert vm._get_next_version_number({'versions': []}) == 1

        # With existing versions
        index = {
            'versions': [
                {'version': 1},
                {'version': 3},
                {'version': 2}
            ]
        }
        assert vm._get_next_version_number(index) == 4

    def test_create_version_metadata(self, vm):
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

    def test_save_version(self, vm, sample_file):
        """Test version saving"""
        file_path, content = sample_file

        # First save
        version = vm.save_version(file_path, content,
                                changes=[{"type": "initial", "description": "First version"}],
                                explanation="Initial version")
        assert version == 1

        # Save same content (should return existing version)
        version2 = vm.save_version(file_path, content)
        assert version2 == 1

        # Save different content
        new_content = "def hello():\n    return 'universe'"
        version3 = vm.save_version(file_path, new_content)
        assert version3 == 2

        # Verify files exist
        meta_dir = vm._get_meta_dir(file_path)
        assert os.path.exists(os.path.join(meta_dir, "index.json"))
        assert os.path.exists(os.path.join(meta_dir, "sample_v1.py"))
        assert os.path.exists(os.path.join(meta_dir, "sample_v2.py"))

    def test_save_original(self, vm, sample_file):
        """Test saving original version"""
        file_path, content = sample_file

        # First call should save the original
        version = vm.save_original(file_path, content)
        assert version == 1

        # Second call should return 1 (already exists)
        version2 = vm.save_original(file_path, content)
        assert version2 == 1

    def test_save_test_version(self, vm, temp_dir):
        """Test saving test version"""
        # Create main file
        main_file = os.path.join(temp_dir, "module.py")
        main_content = "def main(): pass"
        with open(main_file, 'w') as f:
            f.write(main_content)

        # Create test file
        test_file = os.path.join(temp_dir, "module_test.py")
        test_content = "def test_main(): assert True"

        # Save test version
        version = vm.save_test_version(test_file, test_content, main_file_path=main_file)
        assert version == 1

        # Verify test meta directory structure
        main_meta_dir = vm._get_meta_dir(main_file)
        test_meta_dir = vm._get_test_meta_dir(test_file, main_file)
        assert os.path.exists(test_meta_dir)
        assert test_meta_dir.startswith(main_meta_dir)

    def test_save_analysis(self, vm, sample_file):
        """Test analysis saving"""
        file_path, _ = sample_file

        analysis_path = vm.save_analysis(
            file_path=file_path,
            analysis="Code looks good",
            test_path="/test/test_file.py",
            exit_code=0,
            output="All tests passed"
        )

        assert os.path.exists(analysis_path)

        # Load and verify
        with open(analysis_path, 'r') as f:
            data = json.load(f)

        assert data['analysis'] == "Code looks good"
        assert data['test_path'] == "/test/test_file.py"
        assert data['exit_code'] == 0
        assert data['output'] == "All tests passed"

    def test_load_analysis(self, vm, sample_file):
        """Test analysis loading"""
        file_path, _ = sample_file

        # No analysis exists
        result = vm._load_analysis(file_path)
        assert result is None

        # Save and load analysis
        vm.save_analysis(file_path, "Test analysis")
        result = vm._load_analysis(file_path)

        assert result is not None
        assert result['analysis'] == "Test analysis"

    def test_format_analysis_context(self, vm):
        """Test analysis context formatting"""
        # Empty data
        assert vm._format_analysis_context(None) == ""
        assert vm._format_analysis_context({}) == ""

        # With data
        analysis_data = {
            'output': 'Test output',
            'analysis': 'Test analysis'
        }

        result = vm._format_analysis_context(analysis_data)
        assert 'Test output' in result
        assert 'Test analysis' in result
        assert 'IMPORTANT' in result

    def test_get_analysis_prompt(self, vm, sample_file):
        """Test analysis prompt generation"""
        file_path, _ = sample_file

        # No analysis
        result = vm.get_analysis_prompt(file_path)
        assert result is None

        # With analysis
        vm.save_analysis(file_path, "Test analysis", output="Test output")
        result = vm.get_analysis_prompt(file_path)

        assert result is not None
        assert 'Test analysis' in result
        assert 'Test output' in result

    def test_infer_main_file_from_test(self, vm, temp_dir):
        """Test main file inference from test file"""
        # Create main file
        main_file = os.path.join(temp_dir, "module.py")
        with open(main_file, 'w') as f:
            f.write("def main(): pass")

        # Test file that should match
        test_file = os.path.join(temp_dir, "module_test.py")
        result = vm._infer_main_file_from_test(test_file)
        assert result == main_file

        # Test file that shouldn't match
        other_test = os.path.join(temp_dir, "other_test.py")
        result = vm._infer_main_file_from_test(other_test)
        assert result is None

        # Non-test file
        regular_file = os.path.join(temp_dir, "regular.py")
        result = vm._infer_main_file_from_test(regular_file)
        assert result is None

    def test_flag_active(self, vm, sample_file, mock_agent):
        """Test active flagging"""
        file_path, _ = sample_file

        # Mock the process checking to simulate a running process
        with patch.object(vm, '_is_process_running', return_value=True):
            # Flag as active
            result = vm.flag_active(mock_agent, file_path)
            assert result is True

            # Check if flagged
            assert vm.check_active(file_path) is True

    def test_check_active(self, vm, sample_file):
        """Test active checking"""
        file_path, _ = sample_file

        # Not active initially
        assert vm.check_active(file_path) is False

    def test_active_with_stale_process(self, vm, sample_file, mock_agent):
        """Test active checking with stale process (no mocking)"""
        file_path, _ = sample_file

        # Flag as active (this will succeed)
        result = vm.flag_active(mock_agent, file_path)
        assert result is True

        # Check active without mocking - should return False because
        # the mock agent's PID doesn't correspond to a real running process
        assert vm.check_active(file_path) is False

        # Verify the active file was cleaned up due to stale process
        meta_dir = vm._get_meta_dir(file_path)
        active_file_path = os.path.join(meta_dir, "active.json")
        assert not os.path.exists(active_file_path)

    def test_unflag_active(self, vm, sample_file, mock_agent):
        """Test active unflagging"""
        file_path, _ = sample_file

        # Mock the process checking to simulate a running process
        with patch.object(vm, '_is_process_running', return_value=True):
            # Flag first
            vm.flag_active(mock_agent, file_path)
            assert vm.check_active(file_path) is True

            # Unflag
            result = vm.unflag_active(mock_agent, file_path)
            assert result is True
            assert vm.check_active(file_path) is False

            # Try to unflag again
            result = vm.unflag_active(mock_agent, file_path)
            assert result is False

    def test_check_finished(self, vm, sample_file, mock_agent):
        """Test finished checking"""
        file_path, _ = sample_file

        # Not finished initially
        assert vm.check_finished(mock_agent, file_path) is False

    def test_flag_finished(self, vm, sample_file, mock_agent):
        """Test finished flagging"""
        file_path, _ = sample_file

        # Flag as finished
        vm.flag_finished(mock_agent, file_path)
        assert vm.check_finished(mock_agent, file_path) is True

        # Flag again (should not duplicate)
        vm.flag_finished(mock_agent, file_path)
        assert vm.check_finished(mock_agent, file_path) is True

    def test_edge_cases(self, vm, temp_dir):
        """Test edge cases and error conditions"""
        # Test with empty content
        file_path = os.path.join(temp_dir, "empty.py")
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

    def test_file_lock_context_manager(self, vm, temp_dir):
        """Test file locking context manager"""
        lock_file = os.path.join(temp_dir, "test.lock")

        # Test that context manager works without errors
        try:
            with vm._file_lock(lock_file):
                assert os.path.exists(lock_file)
        except Exception as e:
            pytest.fail(f"File lock context manager failed: {e}")

    def test_integration_workflow(self, vm, sample_file, mock_agent):
        """Test complete integration workflow"""
        file_path, content = sample_file

        # Complete workflow test
        try:
            # Mock the process checking to simulate a running process
            with patch.object(vm, '_is_process_running', return_value=True):
                # 1. Save version
                version = vm.save_version(file_path, content)
                assert version >= 1

                # 2. Save analysis
                analysis_path = vm.save_analysis(file_path, "Analysis complete")
                assert os.path.exists(analysis_path)

                # 3. Flag as active
                flag_result = vm.flag_active(mock_agent, file_path)
                assert flag_result is True

                # 4. Check active
                is_active = vm.check_active(file_path)
                assert is_active is True

                # 5. Unflag active
                unflag_result = vm.unflag_active(mock_agent, file_path)
                assert unflag_result is True

                # 6. Flag as finished
                vm.flag_finished(mock_agent, file_path)
                assert vm.check_finished(mock_agent, file_path) is True

        except Exception as e:
            pytest.fail(f"Integration workflow failed: {e}")

    def test_robust_error_handling(self, vm, temp_dir):
        """Test robust error handling"""
        # Test with invalid paths
        try:
            result = vm._load_json_safely("/nonexistent/path/file.json")
            assert result == {}
        except Exception as e:
            pytest.fail(f"Should handle invalid paths gracefully: {e}")

        # Test with permission issues (simulate)
        try:
            vm._get_meta_dir(os.path.join(temp_dir, "test_file.py"))
        except Exception as e:
            pytest.fail(f"Should handle directory creation gracefully: {e}")

    def test_concurrent_access_simulation(self, vm, sample_file, mock_agent):
        """Test simulation of concurrent access patterns"""
        file_path, content = sample_file

        # Multiple versions in sequence
        for i in range(5):
            modified_content = content + f"\n# Version {i}"
            version = vm.save_version(file_path, modified_content)
            assert version == i + 1

        # Verify index integrity
        meta_dir = vm._get_meta_dir(file_path)
        index_path = os.path.join(meta_dir, "index.json")
        with open(index_path, 'r') as f:
            index = json.load(f)

        assert len(index['versions']) == 5
        versions = [v['version'] for v in index['versions']]
        assert versions == [1, 2, 3, 4, 5]

    def test_process_utilities(self, vm):
        """Test process-related utility methods"""
        # Test PID extraction from agent ID
        agent_id = "test_agent_12345"
        pid = vm._get_pid_from_agent_id(agent_id)
        assert pid == 12345

        # Test invalid agent ID
        invalid_id = "invalid_format"
        pid = vm._get_pid_from_agent_id(invalid_id)
        assert pid is None

        # Test process running check with mock
        with patch('psutil.Process') as mock_process:
            mock_proc = mock_process.return_value
            mock_proc.username.return_value = "testuser"
            mock_proc.name.return_value = "python"
            mock_proc.cmdline.return_value = ["python", "script.py"]

            with patch('os.getlogin', return_value="testuser"):
                result = vm._is_process_running(12345)
                assert isinstance(result, bool)


def run_tests():
    """Run all tests with proper setup"""
    print("Testing VersionManager (real implementation)")
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
        print("🎉 Real VersionManager implementation tested successfully!")
    else:
        print("❌ Some tests failed!")
        print("💡 Check the error messages above for details")

    return exit_code


if __name__ == "__main__":
    # Allow running with: python test_version_manager.py
    run_tests()
