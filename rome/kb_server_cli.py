#!/usr/bin/env python3
"""
ChromaDB Server Command Line Interface

Usage:
    python kb_server_cli.py start [--host HOST] [--port PORT] [--path PATH] [--force]
    python kb_server_cli.py stop [--force]
    python kb_server_cli.py restart [--host HOST] [--port PORT] [--path PATH]
    python kb_server_cli.py status
    python kb_server_cli.py clear [--collection NAME] [--force]
    python kb_server_cli.py list [--collection NAME] [--limit LIMIT]
"""

import argparse
import os
import sys
import json
import traceback
from typing import Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rome.kb_server import ChromaServerManager
from rome.logger import get_logger
from rome.config import LONG_SUMMARY_LEN

logger = get_logger()
logger.configure({
    "level": "DEBUG",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "console": True,
    "include_caller_info": "rome"
})

DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 8000
DEFAULT_PATH = None
SERVER_CONFIG = {
    'host': DEFAULT_HOST,
    'port': DEFAULT_PORT,
    'persist_path': DEFAULT_PATH
}


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="ChromaDB Server Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start server with defaults
    python kb_server_cli.py start

    # Start server on specific host/port with custom data path
    python kb_server_cli.py start --host 0.0.0.0 --port 8001 --path ./my_chroma_data

    # Stop server gracefully
    python kb_server_cli.py stop

    # Force stop server (kills process)
    python kb_server_cli.py stop --force

    # Restart server
    python kb_server_cli.py restart

    # Check server status and database info
    python kb_server_cli.py status

    # Clear all data (requires confirmation)
    python kb_server_cli.py clear --force

    # Clear specific collection
    python kb_server_cli.py clear --collection my_collection

    # List all documents in all collections
    python kb_server_cli.py list

    # List documents in specific collection
    python kb_server_cli.py list --collection my_collection

    # List first 10 documents only
    python kb_server_cli.py list --limit 10
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Start command
    start_parser = subparsers.add_parser('start', help='Start ChromaDB server')
    start_parser.add_argument('--host', default=DEFAULT_HOST, help='Server host (default: localhost)')
    start_parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='Server port (default: 8000)')
    start_parser.add_argument('--path', type=str, default=DEFAULT_PATH, help='Data persistence path (default: auto-detect)')
    start_parser.add_argument('--force', action='store_true', help='Force restart if already running')
    start_parser.add_argument('--detach', action='store_true', default=True, help='Run server in background (default: True)')

    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop ChromaDB server')
    stop_parser.add_argument('--force', action='store_true', help='Force kill server process')

    # Restart command
    restart_parser = subparsers.add_parser('restart', help='Restart ChromaDB server')
    restart_parser.add_argument('--host', default=DEFAULT_HOST, help='Server host (default: localhost)')
    restart_parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='Server port (default: 8000)')
    restart_parser.add_argument('--path', type=str, default=DEFAULT_PATH, help='Data persistence path (default: auto-detect)')

    # Status command (now includes database info)
    subparsers.add_parser('status', help='Show server status and database information')

    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear database or collection')
    clear_parser.add_argument('--collection', help='Clear specific collection (default: clear all)')
    clear_parser.add_argument('--force', action='store_true', help='Skip confirmation prompt')

    # List command (new)
    list_parser = subparsers.add_parser('list', help='List documents in collections')
    list_parser.add_argument('--collection', help='List documents in specific collection (default: all collections)')
    list_parser.add_argument('--limit', type=int, default=None, help='Limit number of documents to display')

    return parser


def get_server_manager() -> 'ChromaServerManager':
    """Create ChromaServerManager instance with CLI arguments"""
    return ChromaServerManager.get_instance(SERVER_CONFIG)


def print_status_and_info(status: Dict, db_info: Dict):
    """Print combined status and database information"""
    # Server status section
    if status['running']:
        print(f"✅ ChromaDB server is RUNNING")
        print(f"   URL: {status['url']}")
        print(f"   PID: {status['process_id']}")
        print(f"   Active clients: {status['active_clients']}")
        print(f"   Host: {status['host']}")
        print(f"   Port: {status['port']}")
        print(f"   Startup timeout: {status['startup_timeout']}s")
        print(f"   Shutdown timeout: {status['shutdown_timeout']}s")
    else:
        print(f"❌ ChromaDB server is NOT RUNNING")
        print(f"   Would run at: {status['url']}")

    # Database info section
    print(f"\n📊 Database Information")
    print(f"   Data path: {db_info['persist_path']}")
    print(f"   Running: {'Yes' if db_info['running'] else 'No'}")

    if 'error' in db_info:
        print(f"   Error: {db_info['error']}")
    elif 'collections' in db_info:
        print(f"   Total collections: {db_info['total_collections']}")

        if db_info['collections']:
            print(f"   Collections:")
            for col in db_info['collections']:
                print(f"     - {col['name']} ({col['count']} documents)")
        else:
            print(f"   Collections: None")


def print_documents(documents: Dict, collection_name: str = None, limit: int = None):
    """Print formatted document information"""
    if collection_name:
        print(f"📄 Documents in collection '{collection_name}':")
    else:
        print(f"📄 All documents:")

    total_docs = 0
    displayed_docs = 0

    for col_name, docs in documents.items():
        if collection_name and col_name != collection_name:
            continue

        doc_count = len(docs)
        total_docs += doc_count

        if not collection_name:
            print(f"\n   Collection: {col_name} ({doc_count} documents)")

        for i, doc in enumerate(docs):
            if limit and displayed_docs >= limit:
                print(f"   ... (showing {limit} of {total_docs} total documents)")
                return

            # Extract document info
            doc_id = doc.get('id', 'Unknown')
            content = doc.get('document', '')
            metadata = doc.get('metadata', {})

            # Truncate content for display
            maxlen = LONG_SUMMARY_LEN
            content_preview = content[:maxlen] + "..." if len(content) > maxlen else content

            print(f"   [{displayed_docs + 1}] ID: {doc_id}")
            print(f"       Content: {content_preview}")
            if metadata:
                print(f"       Metadata: {metadata}")
            print()

            displayed_docs += 1

    if displayed_docs == 0:
        if collection_name:
            print(f"   No documents found in collection '{collection_name}'")
        else:
            print(f"   No documents found in any collection")
    elif not limit or displayed_docs < limit:
        print(f"   Total: {displayed_docs} documents")


def confirm_action(message: str) -> bool:
    """Ask for user confirmation"""
    try:
        response = input(f"{message} (y/N): ").strip().lower()
        return response in ['y', 'yes']
    except KeyboardInterrupt:
        print("\nCancelled.")
        return False


class DetachedServerManager:
    """Wrapper for ChromaServerManager that prevents atexit cleanup"""

    def __init__(self, config: Dict = None):
        # Temporarily disable atexit registration
        original_register = None
        import atexit

        # Monkey patch atexit.register to prevent registration during server creation
        def no_register(*args, **kwargs):
            pass

        original_register = atexit.register
        atexit.register = no_register

        try:
            self.manager = get_server_manager()
            # Clear the shutdown registration flag so it doesn't get registered later
            self.manager._shutdown_registered = True
        finally:
            # Restore original atexit.register
            atexit.register = original_register

    def __getattr__(self, name):
        """Delegate all other attributes to the manager"""
        return getattr(self.manager, name)


def handle_start(args):
    """Handle start command"""
    try:
        # Use detached manager to prevent automatic shutdown
        manager = DetachedServerManager({
            'host': args.host,
            'port': args.port,
            'persist_path': args.path
        })

        if manager.is_running() and not args.force:
            print("✅ ChromaDB server is already running")
            status = manager.get_status()
            db_info = manager.get_database_info()
            print_status_and_info(status, db_info)
            return 0

        print("🚀 Starting ChromaDB server...")
        if args.force:
            print("   Force restart requested")

        success = manager.start(force_restart=args.force)

        if success:
            print("✅ ChromaDB server started successfully")
            status = manager.get_status()
            db_info = manager.get_database_info()
            print_status_and_info(status, db_info)

            if args.detach:
                print("🔗 Server running in background. Use 'python kb_server_cli.py stop' to stop.")

            return 0
        else:
            print("❌ Failed to start ChromaDB server")
            print(f"   Try manually: chroma run --host {manager.host} --port {manager.port} --path {manager.persist_path}")
            return 1

    except Exception as e:
        print(f"❌ Error starting server: {e}")
        traceback.print_exc()
        return 1


def handle_stop(args):
    """Handle stop command"""
    try:
        manager = get_server_manager()

        if not manager.is_running():
            print("ℹ️  ChromaDB server is not running")
            return 0

        print("🛑 Stopping ChromaDB server...")
        if args.force:
            print("   Force stop requested")

        success = manager.stop(force=args.force)

        if success:
            print("✅ ChromaDB server stopped successfully")
            return 0
        else:
            print("❌ Failed to stop ChromaDB server")
            return 1

    except Exception as e:
        print(f"❌ Error stopping server: {e}")
        traceback.print_exc()
        return 1


def handle_restart(args):
    """Handle restart command"""
    try:
        manager = get_server_manager()

        print("🔄 Restarting ChromaDB server...")
        success = manager.restart()

        if success:
            print("✅ ChromaDB server restarted successfully")
            status = manager.get_status()
            db_info = manager.get_database_info()
            print_status_and_info(status, db_info)
            return 0
        else:
            print("❌ Failed to restart ChromaDB server")
            return 1

    except Exception as e:
        print(f"❌ Error restarting server: {e}")
        traceback.print_exc()
        return 1


def handle_status(args):
    """Handle status command (now includes database info)"""
    try:
        manager = get_server_manager()
        status = manager.get_status()
        db_info = manager.get_database_info()

        print_status_and_info(status, db_info)
        return 0

    except Exception as e:
        print(f"❌ Error getting status: {e}")
        traceback.print_exc()
        return 1


def handle_clear(args):
    """Handle clear command"""
    try:
        manager = get_server_manager()

        if args.collection:
            # Clear specific collection
            if not args.force:
                if not confirm_action(f"Clear collection '{args.collection}'?"):
                    print("Cancelled.")
                    return 0

            print(f"🗑️  Clearing collection '{args.collection}'...")
            success = manager.clear_collection(args.collection)

            if success:
                print(f"✅ Collection '{args.collection}' cleared successfully")
                return 0
            else:
                print(f"❌ Failed to clear collection '{args.collection}'")
                return 1
        else:
            # Clear entire database
            if not args.force:
                if not confirm_action("Clear ALL database data? This cannot be undone!"):
                    print("Cancelled.")
                    return 0

            print("🗑️  Clearing entire database...")
            success = manager.clear_database(force=True)

            if success:
                print("✅ Database cleared successfully")
                return 0
            else:
                print("❌ Failed to clear database")
                return 1

    except Exception as e:
        print(f"❌ Error clearing data: {e}")
        traceback.print_exc()
        return 1


def handle_list(args):
    """Handle list command - list documents in collections"""
    try:
        manager = get_server_manager()

        if not manager.is_running():
            print("❌ ChromaDB server is not running")
            print("   Use 'python kb_server_cli.py start' to start the server first")
            return 1

        # Get ChromaDB client
        import chromadb
        client = chromadb.HttpClient(host=manager.host, port=manager.port)

        # Get collections
        collections = client.list_collections()

        if not collections:
            print("📄 No collections found in database")
            return 0

        # Collect documents from collections
        documents = {}

        for collection in collections:
            col_name = collection.name

            # Skip if specific collection requested and this isn't it
            if args.collection and col_name != args.collection:
                continue

            try:
                # Get all documents from collection
                result = collection.get(include=['documents', 'metadatas'])

                # Format documents
                col_documents = []
                for i in range(len(result['ids'])):
                    doc = {
                        'id': result['ids'][i],
                        'document': result['documents'][i] if result['documents'] else '',
                        'metadata': result['metadatas'][i] if result['metadatas'] else {}
                    }
                    col_documents.append(doc)

                documents[col_name] = col_documents

            except Exception as e:
                print(f"⚠️  Error reading collection '{col_name}': {e}")
                continue

        if not documents:
            if args.collection:
                print(f"📄 Collection '{args.collection}' not found or empty")
            else:
                print("📄 No documents found in any collection")
            return 0

        # Print documents
        print_documents(documents, args.collection, args.limit)
        return 0

    except Exception as e:
        print(f"❌ Error listing documents: {e}")
        traceback.print_exc()
        return 1


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Only set SERVER_CONFIG for commands that have these arguments
    global SERVER_CONFIG
    if hasattr(args, 'host'):
        SERVER_CONFIG['host'] = args.host
    if hasattr(args, 'port'):
        SERVER_CONFIG['port'] = args.port
    if hasattr(args, 'path'):
        SERVER_CONFIG['persist_path'] = args.path

    # Command dispatch
    commands = {
        'start': handle_start,
        'stop': handle_stop,
        'restart': handle_restart,
        'status': handle_status,
        'clear': handle_clear,
        'list': handle_list,  # New command
    }

    handler = commands.get(args.command)
    if not handler:
        print(f"❌ Unknown command: {args.command}")
        parser.print_help()
        return 1

    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
