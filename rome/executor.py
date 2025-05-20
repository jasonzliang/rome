#!/usr/bin/env python3
# Standalone Command Line Code Executor
# Simplified version with core functionality only

import os
import re
import subprocess
import sys
import platform
import shutil
from hashlib import md5
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Union

from .config import set_attributes_from_config
from .logger import get_logger

# Constants
WIN32 = platform.system() == "Windows"
PYTHON_VARIANTS = ["python", "py", "python3"]
PYTEST_VARIANTS = ["pytest", "py.test"]
TIMEOUT_MSG = "\nExecution timed out."


class CodeBlock:
    """Represents a block of code with its language."""
    def __init__(self, code: str, language: str):
        self.code = code
        self.language = language


class CommandLineCodeResult:
    """Result of a command line code execution."""
    def __init__(self, exit_code: int, output: str, code_file: Optional[str] = None):
        self.exit_code = exit_code
        self.output = output
        self.code_file = code_file


def _cmd(lang: str) -> str:
    """Get the command to execute for a language."""
    if lang == "python":
        return "python" if WIN32 else "python3"
    if lang == "pytest":
        return "pytest"
    if lang in ["bash", "shell", "sh"]:
        return "bash" if not WIN32 else "powershell"
    if lang in ["pwsh", "powershell", "ps1"]:
        return "powershell" if WIN32 else "pwsh"
    return lang


def _get_file_name_from_content(code: str, work_dir: Path) -> Optional[str]:
    """Extract filename from code comments."""
    file_pattern = r"#\s*filename:\s*([^\n]+)"
    match = re.search(file_pattern, code)
    if match:
        filename = match.group(1).strip()
        # Ensure filename is safe and within workspace
        if os.path.isabs(filename):
            normalized_path = os.path.normpath(filename)
            work_dir_path = os.path.normpath(str(work_dir))
            if not normalized_path.startswith(work_dir_path):
                raise ValueError(f"Filename cannot be outside of workspace: {filename}")
        return filename
    return None


def silence_pip(code: str, lang: str) -> str:
    """Modify pip commands to be quiet."""
    if lang != "python":
        return code

    # Replace pip install with quiet flag
    pattern = r"(pip(?:3)?\s+install)"
    replacement = r"\1 -q"
    return re.sub(pattern, replacement, code)


def create_virtual_env(venv_path: Union[str, Path], with_pip: bool = True, install_pytest: bool = True) -> SimpleNamespace:
    """Create a virtual environment at the specified path."""
    import subprocess

    venv_path = Path(venv_path) if isinstance(venv_path, str) else venv_path

    if not venv_path.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv_path), "--with-pip" if with_pip else ""], check=True)

    bin_path = venv_path / ("Scripts" if WIN32 else "bin")
    python_exe = bin_path / ("python.exe" if WIN32 else "python")

    # Install pytest in the virtual environment if requested
    if install_pytest and with_pip:
        try:
            subprocess.run(
                [str(python_exe), "-m", "pip", "install", "pytest"],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install pytest in virtual environment: {e.stderr}")

    return SimpleNamespace(
        bin_path=bin_path,
        env_exe=python_exe,
        env_dir=venv_path,
    )


class CodeExecutor:
    """A code executor that executes or saves code in a local command line environment."""

    SUPPORTED_LANGUAGES = [
        "bash", "shell", "sh", "pwsh", "powershell", "ps1",
        "python", "javascript", "html", "css", "pytest"
    ]

    DEFAULT_EXEC_POLICIES: {
        "bash": True, "shell": True, "sh": True,
        "pwsh": True, "powershell": True, "ps1": True,
        "python": True, "javascript": False, "html": False, "css": False,
        "pytest": True
    }


    def __init__(self, config: Dict = None):
        """Initialize the command line code executor.

        Args:
            config: Configuration dictionary with the following keys:
                - timeout: The timeout for code execution (seconds)
                - virtual_env_context: The virtual environment context
                - execution_policies: Language to execution policies mapping
        """
        # Initialize logger
        self.logger = get_logger()

        # Set default configuration
        self.config = config or {}

        # Update with provided config if any
        set_attributes_from_config(self, self.config, ['timeout', 'virtual_env_context', "work_dir"])

        # Validate timeout
        if self.timeout < 1:
            raise ValueError("Timeout must be greater than or equal to 1.")

        self.logger.debug(f"Initialized {self.__class__.__name__} with config: {merged_config}")

    @staticmethod
    def sanitize_command(lang: str, code: str) -> None:
        """Sanitize the code block to prevent dangerous commands."""
        dangerous_patterns = [
            (r"\brm\s+-rf\b", "Use of 'rm -rf' command is not allowed."),
            (r"\bmv\b.*?\s+/dev/null", "Moving files to /dev/null is not allowed."),
            (r"\bdd\b", "Use of 'dd' command is not allowed."),
            (r">\s*/dev/sd[a-z][1-9]?", "Overwriting disk blocks directly is not allowed."),
            (r":\(\)\{\s*:\|\:&\s*\};:", "Fork bombs are not allowed."),
        ]

        if lang in ["bash", "shell", "sh"]:
            for pattern, message in dangerous_patterns:
                if re.search(pattern, code):
                    raise ValueError(f"Potentially dangerous command detected: {message}")

    def execute_code_blocks(self,
        code_blocks: List[CodeBlock],
        work_dir: Optional[Union[Path, str]] = None) -> CommandLineCodeResult:
        """Execute the code blocks and return the result.

        Args:
            code_blocks: List of code blocks to execute
            work_dir: The working directory for code execution

        Returns:
            Result of the execution
        """
        if not work_dir:
            work_dir = self.work_dir
        if isinstance(work_dir, str):
            work_dir = Path(work_dir)

        # Create work directory if it doesn't exist
        work_dir.mkdir(exist_ok=True)

        logs_all = ""
        file_names = []
        exitcode = 0

        self.logger.debug(f"Executing {len(code_blocks)} code blocks in {work_dir}")

        for code_block in code_blocks:
            lang, code = code_block.language, code_block.code
            lang = lang.lower()

            LocalCommandLineCodeExecutor.sanitize_command(lang, code)
            code = silence_pip(code, lang)

            if lang in PYTHON_VARIANTS:
                lang = "python"

            if lang in PYTEST_VARIANTS:
                lang = "pytest"

            if WIN32 and lang in ["sh", "shell"]:
                lang = "ps1"

            if lang not in self.SUPPORTED_LANGUAGES:
                # In case the language is not supported, we return an error message.
                exitcode = 1
                logs_all += "\n" + f"unknown language {lang}"
                self.logger.error(f"Unknown language: {lang}")
                break

            execute_code = self.execution_policies.get(lang, False)

            try:
                # Check if there is a filename comment
                filename = _get_file_name_from_content(code, work_dir)
            except ValueError as e:
                self.logger.error(f"Invalid filename: {str(e)}")
                return CommandLineCodeResult(exit_code=1, output="Filename is not in the workspace")

            if filename is None:
                # Create a file with an automatically generated name
                code_hash = md5(code.encode()).hexdigest()
                filename = f"tmp_code_{code_hash}.{'py' if lang.startswith('python') else lang}"

            written_file = (work_dir / filename).resolve()
            with written_file.open("w", encoding="utf-8") as f:
                f.write(code)
            file_names.append(written_file)

            self.logger.debug(f"Saved code to {written_file}")

            if not execute_code:
                # Just return a message that the file is saved.
                logs_all += f"Code saved to {written_file!s}\n"
                continue

            program = _cmd(lang)

            # Handle pytest differently - run it with file or specific test
            if lang == "pytest":
                test_marker = None
                test_func_match = re.search(r'#\s*pytest:\s*([^\n]+)', code)
                if test_func_match:
                    test_marker = test_func_match.group(1).strip()

                cmd = [program]
                if test_marker:
                    cmd.extend(["-xvs", f"{written_file}::{test_marker}"])
                else:
                    cmd.extend(["-xvs", str(written_file.absolute())])
            else:
                cmd = [program, str(written_file.absolute())]
            env = os.environ.copy()

            if self.virtual_env_context:
                virtual_env_abs_path = os.path.abspath(self.virtual_env_context.bin_path)
                path_with_virtualenv = rf"{virtual_env_abs_path}{os.pathsep}{env['PATH']}"
                env["PATH"] = path_with_virtualenv
                if WIN32:
                    activation_script = os.path.join(virtual_env_abs_path, "activate.bat")
                    cmd = [activation_script, "&&", *cmd]

            self.logger.debug(f"Executing command: {cmd}")
            try:
                result = subprocess.run(
                    cmd,
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=float(self.timeout),
                    env=env,
                    encoding="utf-8",
                )
                logs_all += result.stderr
                logs_all += result.stdout
                exitcode = result.returncode

                if exitcode != 0:
                    self.logger.error(f"Command exited with non-zero code: {exitcode}")

            except subprocess.TimeoutExpired:
                logs_all += "\n" + TIMEOUT_MSG
                # Same exit code as the timeout command on linux.
                exitcode = 124
                self.logger.error(f"Command execution timed out after {self.timeout} seconds")
                break

            if exitcode != 0:
                break

        code_file = str(file_names[0]) if len(file_names) > 0 else None
        return CommandLineCodeResult(exit_code=exitcode, output=logs_all, code_file=code_file)

    @staticmethod
    def _detect_file_language(file_path: Union[str, Path]) -> str:
        """Detect the language of a file based on its extension and content.

        Args:
            file_path: Path to the file

        Returns:
            Detected language as a string
        """
        file_path = Path(file_path) if isinstance(file_path, str) else file_path

        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file extension
        extension = file_path.suffix.lower().lstrip('.')

        # Extension-based detection
        extension_mapping = {
            'py': 'python',
            'js': 'javascript',
            'html': 'html',
            'css': 'css',
            'sh': 'bash',
            'bash': 'bash',
            'ps1': 'powershell'
        }

        if extension in extension_mapping:
            language = extension_mapping[extension]

            # Additional check for pytest files
            if language == 'python':
                # Check if it's a pytest file based on naming convention
                file_name = file_path.name.lower()
                if file_name.startswith('test_') or file_name.endswith('_test.py') or file_name.endswith('.test.py'):
                    # Read first few lines to confirm it's a pytest file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read(4096)  # Read first 4KB
                        if 'import pytest' in content or '@pytest' in content or 'pytest.fixture' in content:
                            return 'pytest'

            return language

        # Content-based detection for files without recognizable extensions
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(4096)  # Read first 4KB

            if '#!/usr/bin/env python' in content or 'import ' in content:
                # Check if it's a pytest file
                if 'import pytest' in content or '@pytest' in content or 'pytest.fixture' in content:
                    return 'pytest'
                return 'python'
            elif '#!/bin/bash' in content or '#!/usr/bin/env bash' in content:
                return 'bash'
            elif '<html' in content.lower():
                return 'html'

        # Default fallback
        return 'python'  # Default to python if we can't determine

    def execute_file(self, file_path: Union[str, Path], language: Optional[str] = None) -> CommandLineCodeResult:
        """Execute a file and return the result.

        Args:
            file_path: Path to the file to execute
            language: Language of the file (auto-detected if None)

        Returns:
            Result of the execution
        """
        file_path = Path(file_path) if isinstance(file_path, str) else file_path

        # Check if file exists
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return CommandLineCodeResult(exit_code=1, output=f"File not found: {file_path}")

        # Use the file's directory as the working directory
        work_dir = file_path.parent

        # Detect language if not provided
        if language is None:
            try:
                language = self._detect_file_language(file_path)
                self.logger.debug(f"Detected language for {file_path}: {language}")
            except Exception as e:
                self.logger.error(f"Error detecting language: {str(e)}")
                return CommandLineCodeResult(exit_code=1, output=f"Error detecting language: {str(e)}")

        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            self.logger.error(f"Error reading file: {str(e)}")
            return CommandLineCodeResult(exit_code=1, output=f"Error reading file: {str(e)}")

        # Create a code block and execute it
        code_block = CodeBlock(code=code, language=language)
        return self.execute_code_blocks([code_block], work_dir)


def main():
    pass
