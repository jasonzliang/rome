#!/usr/bin/env python3
"""
Download experiment results from remote server with wildcard support.

WILDCARD USAGE:
  Quote wildcards to prevent local shell expansion:
    ./download.py 'bert*'        # Downloads all remote dirs starting with 'bert'
    ./download.py '*2024*'       # Downloads all remote dirs containing '2024'
    ./download.py 'exp_[0-9]*'   # Downloads dirs matching pattern exp_0*, exp_1*, etc.

  Without quotes, your local shell expands first (usually wrong):
    ./download.py bert*          # Expands to local files, not remote dirs
"""
import os
import sys
import subprocess
import shutil
import argparse
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    console = Console()
    def print_info(msg): console.print(f"[cyan]ℹ {msg}[/cyan]")
    def print_success(msg): console.print(f"[green]✓ {msg}[/green]")
    def print_warning(msg): console.print(f"[yellow]⚠ {msg}[/yellow]")
    def print_error(msg): console.print(f"[red]✗ {msg}[/red]", file=sys.stderr)
    def print_command(msg): console.print(f"[blue]Executing:[/blue] {msg}")
except ImportError:
    console = None
    def print_info(msg): print(f"ℹ {msg}")
    def print_success(msg): print(f"✓ {msg}")
    def print_warning(msg): print(f"⚠ {msg}")
    def print_error(msg): print(f"✗ {msg}", file=sys.stderr)
    def print_command(msg): print(f"Executing: {msg}")

# Configuration
SERVER_IP = "biggpu"
SERVER_DIR = "~/Desktop/rome/benchmark/result"
EXCLUDE_DIRS = []
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

def run_rsync(server_ip, server_dir, local_dir, exclude_dirs, dry_run=False, progress=True):
    """Execute rsync with error handling and validation."""
    if not shutil.which("rsync"):
        print_error("rsync not found. Please install rsync.")
        return False

    # Build rsync command - back to the simple approach that worked
    exclude_args = [f"--exclude={x}" for x in exclude_dirs]
    opts = ["-ahvzAPX", "--no-i-r", "--stats"]
    if dry_run: opts.append("--dry-run")
    if progress: opts.append("--progress")

    # Use shell=True to allow remote wildcard expansion (this actually works!)
    exclude_str = " ".join(exclude_args)
    opts_str = " ".join(opts)
    cmd = f"rsync {opts_str} {exclude_str} {server_ip}:{server_dir} {local_dir}"

    if dry_run: print_warning("DRY RUN - No files will be transferred")
    print_command(cmd)

    try:
        if not dry_run: Path(local_dir).mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, shell=True, check=True)
        print_success("Dry run completed" if dry_run else "Transfer completed")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"rsync failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print_warning("Transfer interrupted by user")
        return False
    except Exception as e:
        print_error(str(e))
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Download experiment results from remote server",
        epilog="""
WILDCARD EXAMPLES:
  ./download.py 'bert*'              # All dirs starting with 'bert'
  ./download.py '*2024*'             # All dirs containing '2024'
  ./download.py 'exp_[0-9]*'         # Dirs like exp_0*, exp_1*, etc.

⚠ Always quote wildcards to prevent local shell expansion!
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("experiment", help="Experiment name or pattern")
    parser.add_argument("server", nargs="?", default=SERVER_IP, help=f"Server IP (default: {SERVER_IP})")
    parser.add_argument("-n", "--dry-run", action="store_true", help="Show what would be transferred")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--exclude", action="append", default=[], help="Exclude pattern")
    parser.add_argument("--local-dir", default=LOCAL_DIR, help="Local download directory")
    parser.add_argument("--no-wildcard", action="store_true", help="Don't auto-append * to experiment name")

    args = parser.parse_args()

    # Auto-append wildcard by default (unless disabled or already has wildcards)
    experiment_name = args.experiment
    if not args.no_wildcard and not any(c in experiment_name for c in ['*', '?', '[']):
        experiment_name += "*"
        print_info(f"Added wildcard: '{args.experiment}' → '{experiment_name}'")

    server_path = os.path.join(SERVER_DIR, experiment_name)

    # Show configuration with rich panel if available
    if console:
        config_text = f"Server: {args.server}\nRemote: {server_path}\nLocal: {args.local_dir}"
        if args.exclude: config_text += f"\nExcludes: {', '.join(args.exclude)}"
        console.print(Panel(config_text, title="[bold]Download Configuration[/bold]", border_style="cyan"))
    else:
        print_info(f"Server: {args.server} | Remote: {server_path} | Local: {args.local_dir}")

    # Execute transfer
    success = run_rsync(args.server, server_path, args.local_dir,
                       EXCLUDE_DIRS + args.exclude, args.dry_run, not args.quiet)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
