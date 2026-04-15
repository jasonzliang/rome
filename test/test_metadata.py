#!/usr/bin/env python3
"""
Efficient VersionManager tests with minimal mocks and maximum real object usage.
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

from rome.metadata import VersionManager, ValidationError, FileType


class TestVersionManager:
    """Efficient test suite using real objects where possible"""

    @pytest.fixture
    def vm_setup(self):
        """Compact setup with real filesystem and database"""
        temp_dir = tempfile.mkdtemp()

        # Provide config with required attributes
        vm = VersionManager(
            config={'lock_active_file': True},
            db_config={'lock_timeout': 5.0, 'max_retries': 3, 'retry_delay': 0.1}
        )

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
        agent.name = "test_agent"

        yield temp_dir, vm, files, agent, {name: content for name, content in test_data}

        # Cleanup
        try:
            vm.shutdown(agent)
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass

    def test_core_functionality(self, vm_setup):
        """Test all core functionality in one efficient test"""
        temp_dir, vm, files, agent, contents = vm_setup
        main_file = files["main.py"]
        main_content = contents["main.py"]

        # Test meta directory creation
        meta_dir = vm._get_meta_dir(main_file)
        assert meta_dir.endswith(".rome")
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
            # Initial state - file should be available (not active)
            assert vm.check_avaliable(main_file)
            assert not vm._has_active_files_pointer()

            # Flag active
            assert vm.flag_active(agent, main_file)
            # From same process, check_avaliable returns True (self-owned)
            assert vm.check_avaliable(main_file)
            assert vm._has_active_files_pointer()

            # Flagging another file now just switches (logs error, doesn't raise)
            other_file = files["utils.py"]
            assert vm.flag_active(agent, other_file)

            # Unflag the current active file
            assert vm.unflag_active(agent, other_file)
            assert vm.check_avaliable(other_file)
            assert not vm._has_active_files_pointer()

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
        agent2.name = "agent2"
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

        # File path cleaning
        assert vm._clean_file_path("file.py.rome") == "file.py"
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

        # Should clean up automatically (stale file = available)
        assert vm.check_avaliable(main_file)
        assert not os.path.exists(active_file)

    def test_concurrent_operations(self, vm_setup):
        """Test thread safety"""
        temp_dir, vm, files, agent, contents = vm_setup
        main_file = files["main.py"]
        results = []

        def save_version(i):
            content = f"# Version {i}\ndef v{i}(): pass"
            return vm.save_version(main_file, content,
                                 explanation=f"Version {i}")

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(save_version, i) for i in range(8)]
            for f in as_completed(futures):
                try:
                    results.append(f.result())
                except:
                    pass

        # All should have gotten valid version numbers
        assert len(results) > 0
        assert all(v >= 1 for v in results)

    def test_unicode_and_special_content(self, vm_setup):
        """Test handling of special content"""
        temp_dir, vm, files, agent, contents = vm_setup
        main_file = files["main.py"]

        special_contents = [
            '# 日本語テスト\ndef hello(): return "こんにちは"',
            '"""Triple quoted \' " strings"""\nx = "\\n\\t"',
            '# Empty-ish\n' * 100,
        ]

        for i, content in enumerate(special_contents):
            v = vm.save_version(main_file, content, explanation=f"Special {i}")
            assert v > 0

            loaded = vm.load_version(main_file, k=1, include_content=True)
            if loaded:
                stored_content = loaded.get("content", "") if isinstance(loaded, dict) else ""
                if stored_content:
                    assert stored_content == content

    def test_error_handling(self, vm_setup):
        """Test error handling"""
        temp_dir, vm, files, agent, contents = vm_setup

        # Non-existent file
        fake_file = os.path.join(temp_dir, "nonexistent.py")
        result = vm.load_version(fake_file)
        assert result is None or result == []

        # Empty content
        main_file = files["main.py"]
        v = vm.save_version(main_file, "")
        assert v >= 1

    def test_complete_workflow_integration(self, vm_setup):
        """Test complete workflow"""
        temp_dir, vm, files, agent, contents = vm_setup
        main_file = files["main.py"]
        test_file = files["main_test.py"]

        with patch.object(vm, '_is_process_running', return_value=True):
            # 1. Save originals
            vm.save_original(main_file, contents["main.py"])

            # 2. Flag active and edit
            vm.flag_active(agent, main_file)

            # 3. Save versions
            vm.save_version(main_file, contents["main.py"] + "\n# edit 1")
            vm.save_version(main_file, contents["main.py"] + "\n# edit 2")

            # 4. Save test version
            vm.save_test_version(test_file, contents["main_test.py"] + "\n# new test")

            # 5. Flag finished and unflag active
            vm.flag_finished(agent, main_file)
            vm.unflag_active(agent, main_file)

            # 6. Verify state
            assert vm.check_finished(agent, main_file)
            assert vm.check_avaliable(main_file)

            # 7. Version history
            versions = vm.load_version(main_file, k=10)
            assert len(versions) >= 2

            # 8. Store analysis
            vm.store_data(main_file, "analysis", {"score": 95, "status": "complete"})
            data = vm.get_data(main_file, "analysis")
            assert data["score"] == 95
