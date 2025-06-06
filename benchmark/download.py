#!/usr/bin/env python3
"""
Download experiment results from remote server with wildcard support.

USAGE:
  ./download.py 'bert*' -s biggpu-vpn -r ~/experiments    # Download matching dirs
  ./download.py --ls 'bert*' -s biggpu-vpn               # List matching dirs
  ./download.py --ls -s biggpu-vpn -r /data/models       # List all dirs

COMMON OPTIONS:
  -s, --server       Server hostname (default: biggpu)
  -r, --remote-dir   Remote directory path (default: ~/Desktop/rome/benchmark/result)
  -l, --local-dir    Local directory path (default: ~/Desktop/rome/benchmark/result)
  -e, --exclude      Exclude pattern (can be used multiple times)
  -w, --no-wildcard  Don't auto-append * to experiment name
  -n, --dry-run      Show what would be transferred without doing it
  --ls               List remote directories instead of downloading

Quote wildcards to prevent local shell expansion!
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns

console = Console()

# Configuration
SERVER_IP = "biggpu"
SERVER_DIR = "~/Desktop/rome/benchmark/result"
LOCAL_DIR = "~/Desktop/rome/benchmark/result"

def run_cmd(cmd, capture=True, timeout=None):
    """Execute command with timeout and error handling."""
    console.print(f"[blue]Executing:[/blue] {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True,
                              capture_output=capture, text=True, timeout=timeout)
        return result.stdout if capture else None, True
    except subprocess.TimeoutExpired:
        console.print(f"[red]✗ Command timed out after {timeout}s[/red]")
        return None, False
    except subprocess.CalledProcessError as e:
        if e.returncode == 255:  # SSH connection error
            console.print(f"[red]✗ Connection failed - check server name and network[/red]")
        else:
            console.print(f"[red]✗ Command failed (exit {e.returncode})[/red]")
        return None, False
    except KeyboardInterrupt:
        console.print("[yellow]⚠ Interrupted by user[/yellow]")
        return None, False

def list_remote(server, remote_dir, pattern="*"):
    """List remote directories matching pattern."""
    cmd = f"ssh {server} 'ls -1dt {remote_dir}/{pattern} 2>/dev/null | head -20'"
    output, success = run_cmd(cmd, timeout=10)

    if not success or not output:
        console.print(f"[yellow]⚠ No directories found matching '{pattern}'[/yellow]")
        return success

    dirs = [os.path.basename(line.strip()) for line in output.strip().split('\n')]
    console.print(f"\n[bold]Remote directories matching '{pattern}':[/bold]")
    console.print(Columns(dirs, equal=True, expand=True))
    console.print(f"[cyan]ℹ Found {len(dirs)} directories[/cyan]")
    return True

def download(server, remote_dir, pattern, local_dir, excludes, dry_run=False):
    """Download files via rsync."""
    # Let shell handle tilde expansion in mkdir
    subprocess.run(f"mkdir -p '{local_dir}'", shell=True, check=True)

    exclude_str = " ".join(f"--exclude={x}" for x in excludes)
    opts = "-ahvzAPX --no-i-r --stats --progress"
    if dry_run: opts += " --dry-run"

    cmd = f"rsync {opts} {exclude_str} {server}:{remote_dir}/{pattern} {local_dir}"

    if dry_run:
        console.print("[yellow]⚠ DRY RUN - No files will be transferred[/yellow]")

    _, success = run_cmd(cmd, capture=False, timeout=300)  # 5 minutes for transfers
    status = "Dry run completed" if dry_run else "Transfer completed"
    console.print(f"[green]✓ {status}[/green]" if success else f"[red]✗ Transfer failed[/red]")
    return success

def main():
    parser = argparse.ArgumentParser(
        description="Download experiment results from remote server",
        epilog="""
WILDCARD EXAMPLES:
  ./download.py 'bert*'              # All dirs starting with 'bert'
  ./download.py '*2024*'             # All dirs containing '2024'
  ./download.py 'exp_[0-9]*'         # Dirs like exp_0*, exp_1*, etc.

REMOTE LISTING:
  ./download.py --ls 'bert*'         # List matching remote directories
  ./download.py --ls                 # List all remote directories

SERVER USAGE:
  ./download.py 'bert*' -s biggpu-vpn     # Download from different server
  ./download.py --ls -s biggpu-vpn        # List from different server
  ./download.py 'bert*' --server myhost   # Long form server option

Always quote wildcards to prevent local shell expansion!
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("experiment", nargs="?", help="Experiment pattern")
    parser.add_argument("-s", "--server", default=SERVER_IP, help=f"Server (default: {SERVER_IP})")
    parser.add_argument("-r", "--remote-dir", default=SERVER_DIR, help=f"Remote directory (default: {SERVER_DIR})")
    parser.add_argument("-n", "--dry-run", action="store_true", help="Dry run")
    parser.add_argument("-e", "--exclude", action="append", default=[], help="Exclude pattern")
    parser.add_argument("-l", "--local-dir", default=LOCAL_DIR, help="Local directory")
    parser.add_argument("-w", "--no-wildcard", action="store_true", help="Don't auto-append *")
    parser.add_argument("--ls", action="store_true", help="List remote directories")

    args = parser.parse_args()

    # Auto-append wildcard unless disabled or already present
    pattern = args.experiment or "*"
    if not args.no_wildcard and args.experiment and not any(c in pattern for c in '*?['):
        pattern += "*"
        console.print(f"[cyan]ℹ Added wildcard: '{args.experiment}' → '{pattern}'[/cyan]")

    # List or download
    if args.ls:
        success = list_remote(args.server, args.remote_dir, pattern)
    else:
        if not args.experiment:
            console.print("[red]✗ Experiment name required for download (use --ls to list)[/red]")
            sys.exit(1)

        # Show config
        config = f"Server: {args.server}\nRemote: {args.remote_dir}/{pattern}\nLocal: {args.local_dir}"
        if args.exclude: config += f"\nExcludes: {', '.join(args.exclude)}"
        console.print(Panel(config, title="[bold]Download Configuration[/bold]", border_style="cyan"))

        success = download(args.server, args.remote_dir, pattern, args.local_dir, args.exclude, args.dry_run)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
