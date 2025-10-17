#!/usr/bin/env python
"""Universal Streamlit launcher with automatic dependency detection and incremental installation."""
import sys
import os
import subprocess
import ast
from pathlib import Path

# Mapping of import names to pip package names
IMPORT_TO_PACKAGE = {
    'sklearn': 'scikit-learn', 'cv2': 'opencv-python', 'PIL': 'Pillow',
    'yaml': 'pyyaml', 'bs4': 'beautifulsoup4', 'MySQLdb': 'mysqlclient',
    'google.cloud': 'google-cloud-storage', 'dateutil': 'python-dateutil',
    'dotenv': 'python-dotenv', 'magic': 'python-magic',
}

def extract_imports(script_path):
    """Extract all imports and pip install comments from a Python script."""
    imports = set()
    pip_packages = set()

    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse pip install comments (e.g., # pip install pandas numpy)
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('#') and 'pip install' in line:
                # Extract package names after "pip install"
                parts = line.split('pip install', 1)
                if len(parts) == 2:
                    # Split by whitespace and filter out flags/options
                    packages = parts[1].strip().split()
                    for pkg in packages:
                        # Skip pip flags (starting with -)
                        if not pkg.startswith('-'):
                            pip_packages.add(pkg)

        # Parse AST for import statements
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except Exception as e:
        print(f"Warning: Could not parse {script_path}: {e}")

    return imports, pip_packages

def get_package_names(imports):
    """Convert import names to pip package names."""
    packages = set()

    # Standard library modules to skip
    stdlib = {
        'sys', 'os', 'subprocess', 'pathlib', 'json', 're', 'collections',
        'multiprocessing', 'warnings', 'ast', 'time', 'datetime', 'math',
        'random', 'itertools', 'functools', 'typing', 'copy', 'pickle',
        'io', 'logging', 'argparse', 'shutil', 'glob', 'tempfile', 'uuid',
        'hashlib', 'base64', 'struct', 'enum', 'dataclasses', 'abc'
    }

    for imp in imports:
        # Skip standard library modules
        if imp in stdlib:
            continue

        # Map import name to package name
        package = IMPORT_TO_PACKAGE.get(imp, imp)
        packages.add(package)

    return sorted(packages)

def get_installed_packages(venv_python):
    """Get list of currently installed packages in the venv."""
    try:
        result = subprocess.run(
            [str(venv_python), "-m", "pip", "list", "--format=freeze"],
            capture_output=True, text=True, check=True)

        installed = set()
        for line in result.stdout.strip().split('\n'):
            if '==' in line:
                # Normalize package name (lowercase, replace _ with -)
                pkg_name = line.split('==')[0].lower().replace('_', '-')
                installed.add(pkg_name)

        return installed
    except Exception as e:
        print(f"Warning: Could not get installed packages: {e}")
        return set()

def normalize_package_name(name):
    """Normalize package name for comparison."""
    return name.lower().replace('_', '-')

def setup_venv(script_path, venv_dir):
    """Create virtualenv and install packages if needed."""
    # Create venv if it doesn't exist
    if not venv_dir.exists():
        print(f"Creating virtualenv at {venv_dir}...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

    venv_python = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python"

    # ALWAYS analyze script for dependencies (this is key!)
    print(f"Analyzing {script_path.name} for dependencies...")
    imports, pip_packages = extract_imports(script_path)

    # Convert imports to package names
    auto_packages = get_package_names(imports)

    # Combine auto-detected and explicit pip install packages
    all_packages = sorted(set(auto_packages) | pip_packages)

    if not all_packages:
        print("No external packages detected.")
        return venv_python

    # Show what was found
    if pip_packages:
        print(f"  From pip comments: {', '.join(sorted(pip_packages))}")
    if auto_packages:
        print(f"  From imports: {', '.join(auto_packages)}")

    # ALWAYS check what's currently installed (this is key!)
    print(f"Checking installed packages...")
    installed = get_installed_packages(venv_python)

    # Normalize all package names for comparison
    installed_normalized = {normalize_package_name(pkg) for pkg in installed}

    # Find packages that need to be installed
    needed = []
    for pkg in all_packages:
        if normalize_package_name(pkg) not in installed_normalized:
            needed.append(pkg)

    if needed:
        print(f"Installing {len(needed)} new package(s): {', '.join(needed)}")

        # Upgrade pip first
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
            check=True, capture_output=True)

        # Install missing packages with binary wheels when possible
        result = subprocess.run([
            str(venv_python), "-m", "pip", "install",
            "--only-binary", ":all:"
        ] + needed, capture_output=True, text=True)

        if result.returncode != 0:
            print("Note: Some packages don't have binary wheels, retrying without --only-binary...")
            subprocess.run([
                str(venv_python), "-m", "pip", "install"
            ] + needed, check=True)

        print(f"✓ Successfully installed {len(needed)} package(s)")
    else:
        print("✓ All required packages already installed")

    return venv_python

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: streamlit_run.py <script.py> [streamlit args...]")
        print("\nExample:")
        print("  ./streamlit_run.py my_app.py")
        print("  ./streamlit_run.py my_app.py --server.port 8502")
        print("\nFeatures:")
        print("  - Auto-detects dependencies from imports")
        print("  - Reads '# pip install ...' comments")
        print("  - Creates isolated venv per app")
        print("  - Only installs missing packages")
        sys.exit(1)

    script_path = Path(sys.argv[1]).resolve()

    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        sys.exit(1)

    # Create app-specific venv under .venv directory
    venv_root = script_path.parent / ".venv"
    venv_root.mkdir(exist_ok=True)

    script_stem = script_path.stem  # filename without extension
    venv_dir = venv_root / script_stem

    print(f"App: {script_path.name}")
    print(f"Virtual environment: .venv/{script_stem}")
    print()

    # Check if already in virtualenv
    if sys.prefix == sys.base_prefix:
        setup_venv(script_path, venv_dir)
        print()
        print("Launching Streamlit app...")
        print("-" * 50)

        # Launch via streamlit CLI
        streamlit_path = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "streamlit"
        os.execv(str(streamlit_path),
                [str(streamlit_path), "run", str(script_path)] + sys.argv[2:])
    else:
        # Already in venv, just run normally (shouldn't reach here usually)
        print("Already in virtualenv")

if __name__ == "__main__":
    main()