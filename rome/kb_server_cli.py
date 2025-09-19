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
    python kb_server_cli.py export [--collection NAME] [--output FILE] [--include-embeddings]
    python kb_server_cli.py import [--file FILE] [--collection NAME] [--overwrite]
"""
import atexit
import argparse
import os
import sys
import json
import traceback
from typing import Dict, Optional, List, Any

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

    # List up to 10 documents in a collection
    python kb_server_cli.py list --collection my_collection --limit 10

    # Export specific collection with embeddings to JSON file
    python kb_server_cli.py export --collection my_collection --include-embeddings --output my_export.json

    # Import from JSON file and overwrite existing collection
    python kb_server_cli.py import --file my_export.json --overwrite

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

    # Status command
    subparsers.add_parser('status', help='Show server status and database information')

    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear database or collection')
    clear_parser.add_argument('--collection', help='Clear specific collection (default: clear all)')
    clear_parser.add_argument('--force', action='store_true', help='Skip confirmation prompt')

    # List command
    list_parser = subparsers.add_parser('list', help='List documents in collections')
    list_parser.add_argument('--collection', help='List documents in specific collection (default: all collections)')
    list_parser.add_argument('--limit', type=int, default=None, help='Limit number of documents to display')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export collections to JSON')
    export_parser.add_argument('--collection', help='Export specific collection (default: all collections)')
    export_parser.add_argument('--output', help='Output JSON file (default: auto-generated)')
    export_parser.add_argument('--include-embeddings', action='store_true', help='Include embeddings in export')
    export_parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for large collections (default: 1000)')

    # Import command
    import_parser = subparsers.add_parser('import', help='Import collections from JSON')
    import_parser.add_argument('--file', required=True, help='JSON file to import')
    import_parser.add_argument('--collection', help='Import to specific collection name (overrides file)')
    import_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing collections')
    import_parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for import (default: 1000)')

    return parser

def get_server_manager() -> 'ChromaServerManager':
    """Create ChromaServerManager instance with CLI arguments"""
    return ChromaServerManager.get_instance(SERVER_CONFIG)

def get_client():
    """Get ChromaDB client"""
    import chromadb
    manager = get_server_manager()
    return chromadb.HttpClient(host=manager.host, port=manager.port)

def export_collection_to_dict(collection, include_embeddings: bool = False, batch_size: int = 1000) -> Dict[str, Any]:
    """Export a single collection to dictionary format"""
    total_count = collection.count()

    export_data = {
        'collection_name': collection.name,
        'count': total_count,
        'data': []
    }

    # Process in batches for memory efficiency
    for offset in range(0, total_count, batch_size):
        include_fields = ['documents', 'metadatas']
        if include_embeddings:
            include_fields.append('embeddings')

        results = collection.get(
            limit=batch_size,
            offset=offset,
            include=include_fields
        )

        for i in range(len(results['ids'])):
            item = {
                'id': results['ids'][i],
                'document': results['documents'][i] if results['documents'] else None,
                'metadata': results['metadatas'][i] if results['metadatas'] else None
            }

            if include_embeddings and results.get('embeddings'):
                item['embedding'] = results['embeddings'][i]

            export_data['data'].append(item)

    return export_data

def import_collection_from_dict(client, data: Dict[str, Any], target_name: str = None, overwrite: bool = False, batch_size: int = 1000) -> bool:
    """Import a collection from dictionary data"""
    collection_name = target_name or data['collection_name']

    # Check if collection exists
    try:
        existing_collections = [col.name for col in client.list_collections()]
        collection_exists = collection_name in existing_collections

        if collection_exists:
            if overwrite:
                print(f"   Deleting existing collection '{collection_name}'")
                client.delete_collection(collection_name)
            else:
                print(f"   Collection '{collection_name}' already exists (use --overwrite to replace)")
                return False
    except Exception as e:
        print(f"   Warning: Could not check existing collections: {e}")

    # Create collection
    try:
        collection = client.create_collection(collection_name)
        print(f"   Created collection '{collection_name}'")
    except Exception as e:
        print(f"   Error creating collection '{collection_name}': {e}")
        return False

    # Import data in batches
    total_items = len(data['data'])
    imported_count = 0

    for start_idx in range(0, total_items, batch_size):
        end_idx = min(start_idx + batch_size, total_items)
        batch_data = data['data'][start_idx:end_idx]

        # Separate batch data
        batch_ids = []
        batch_documents = []
        batch_metadatas = []
        batch_embeddings = []

        has_documents = False
        has_metadatas = False
        has_embeddings = False

        for item in batch_data:
            batch_ids.append(item['id'])

            if item.get('document') is not None:
                batch_documents.append(item['document'])
                has_documents = True
            else:
                batch_documents.append('')

            if item.get('metadata') is not None:
                batch_metadatas.append(item['metadata'])
                has_metadatas = True
            else:
                batch_metadatas.append({})

            if item.get('embedding') is not None:
                batch_embeddings.append(item['embedding'])
                has_embeddings = True

        # Add batch to collection
        try:
            add_kwargs = {'ids': batch_ids}
            if has_documents:
                add_kwargs['documents'] = batch_documents
            if has_metadatas:
                add_kwargs['metadatas'] = batch_metadatas
            if has_embeddings:
                add_kwargs['embeddings'] = batch_embeddings

            collection.add(**add_kwargs)
            imported_count += len(batch_data)

            if total_items > batch_size:
                print(f"   Imported {imported_count}/{total_items} documents...")

        except Exception as e:
            print(f"   Error importing batch {start_idx}-{end_idx}: {e}")
            return False

    print(f"   Successfully imported {imported_count} documents to '{collection_name}'")
    return True

def generate_output_filename(collection_name: str = None) -> str:
    """Generate output filename for export"""
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if collection_name:
        return f"chroma_export_{collection_name}_{timestamp}.json"
    else:
        return f"chroma_export_all_{timestamp}.json"

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
    """Print human-readable formatted document information"""
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
        maxlen = LONG_SUMMARY_LEN

        if not collection_name:
            print(f"\n━━━ Collection: {col_name} ({doc_count} documents) ━━━")

        for i, doc in enumerate(docs):
            if limit and displayed_docs >= limit:
                print(f"\n💡 Showing first {limit} of {total_docs} total documents")
                return

            # Extract document info
            doc_id = doc.get('id', 'Unknown')
            content = doc.get('document', '')
            metadata = doc.get('metadata', {})

            # Format content with smart truncation
            if len(content) > maxlen:
                # Try to break at word boundary
                truncated = content[:maxlen]
                last_space = truncated.rfind(' ')
                if last_space > maxlen * 0.8:  # If space is reasonably close to end
                    content_preview = truncated[:last_space] + "..."
                else:
                    content_preview = truncated + "..."
            else:
                content_preview = content

            # Pretty print document
            print(f"\n🔸 Document #{displayed_docs + 1}")
            print(f"   ID: {doc_id}")

            if content_preview:
                print(f"   Content:")
                # Indent content for better readability
                for line in content_preview.split('\n'):
                    print(f"     {line}")
            else:
                print(f"   Content: (empty)")

            if metadata:
                print(f"   Metadata:")
                for key, value in sorted(metadata.items()):
                    # Format metadata values nicely
                    if isinstance(value, (dict, list)):
                        value_str = json.dumps(value, indent=6)
                        print(f"     {key}: {value_str}")
                    else:
                        print(f"     {key}: {value}")

            displayed_docs += 1

    print(f"\n{'─' * 60}")
    if displayed_docs == 0:
        if collection_name:
            print(f"   No documents found in collection '{collection_name}'")
        else:
            print(f"   No documents found in any collection")
    else:
        if collection_name:
            print(f"   Total: {displayed_docs} documents in '{collection_name}'")
        else:
            collections_shown = len([name for name in documents.keys() if not collection_name or name == collection_name])
            print(f"   Total: {displayed_docs} documents across {collections_shown} collections")

def confirm_action(message: str) -> bool:
    """Ask for user confirmation"""
    try:
        response = input(f"{message} (Y/n): ").strip().lower()
        return response not in ['n', 'no']
    except KeyboardInterrupt:
        print("\nCancelled.")
        return False

class DetachedServerManager:
    """Wrapper for ChromaServerManager that prevents atexit cleanup"""
    def __init__(self, config: Dict = None):
        # Temporarily disable atexit registration
        original_register = None

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
        client = get_client()

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

def handle_export(args):
    """Handle export command - export collections to JSON"""
    try:
        manager = get_server_manager()

        if not manager.is_running():
            print("❌ ChromaDB server is not running")
            print("   Use 'python kb_server_cli.py start' to start the server first")
            return 1

        # Get ChromaDB client
        client = get_client()

        # Get collections
        collections = client.list_collections()
        if not collections:
            print("📄 No collections found in database")
            return 0

        # Generate output filename if not provided
        output_file = args.output or generate_output_filename(args.collection)

        print(f"📤 Exporting to '{output_file}'...")
        if args.include_embeddings:
            print("   Including embeddings in export")

        export_data = {}

        if args.collection:
            # Export specific collection
            target_collection = None
            for col in collections:
                if col.name == args.collection:
                    target_collection = col
                    break

            if not target_collection:
                print(f"❌ Collection '{args.collection}' not found")
                return 1

            print(f"   Exporting collection '{args.collection}'...")
            collection_data = export_collection_to_dict(
                target_collection,
                args.include_embeddings,
                args.batch_size
            )
            export_data = {
                'export_type': 'single_collection',
                'collections': [collection_data]
            }
        else:
            # Export all collections
            export_data = {
                'export_type': 'all_collections',
                'collections': []
            }

            for collection in collections:
                print(f"   Exporting collection '{collection.name}'...")
                collection_data = export_collection_to_dict(
                    collection,
                    args.include_embeddings,
                    args.batch_size
                )
                export_data['collections'].append(collection_data)

        # Write to JSON file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            # Summary
            total_collections = len(export_data['collections'])
            total_documents = sum(col['count'] for col in export_data['collections'])

            print(f"✅ Export completed successfully")
            print(f"   File: {output_file}")
            print(f"   Collections: {total_collections}")
            print(f"   Total documents: {total_documents}")
            print(f"   Embeddings included: {'Yes' if args.include_embeddings else 'No'}")

            return 0

        except Exception as e:
            print(f"❌ Error writing export file: {e}")
            return 1

    except Exception as e:
        print(f"❌ Error during export: {e}")
        traceback.print_exc()
        return 1

def handle_import(args):
    """Handle import command - import collections from JSON"""
    try:
        manager = get_server_manager()

        if not manager.is_running():
            print("❌ ChromaDB server is not running")
            print("   Use 'python kb_server_cli.py start' to start the server first")
            return 1

        # Check if input file exists
        if not os.path.exists(args.file):
            print(f"❌ Input file '{args.file}' not found")
            return 1

        print(f"📥 Importing from '{args.file}'...")

        # Load JSON data
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
        except Exception as e:
            print(f"❌ Error reading import file: {e}")
            return 1

        # Validate import data structure
        if 'collections' not in import_data:
            print("❌ Invalid import file format: missing 'collections' key")
            return 1

        # Get ChromaDB client
        client = get_client()

        # Import collections
        imported_collections = 0
        total_documents = 0

        for collection_data in import_data['collections']:
            if 'collection_name' not in collection_data:
                print("⚠️  Skipping collection with missing name")
                continue

            collection_name = args.collection or collection_data['collection_name']
            print(f"   Importing collection '{collection_name}'...")

            success = import_collection_from_dict(
                client,
                collection_data,
                collection_name,
                args.overwrite,
                args.batch_size
            )

            if success:
                imported_collections += 1
                total_documents += collection_data.get('count', 0)
            else:
                print(f"⚠️  Failed to import collection '{collection_name}'")

        if imported_collections > 0:
            print(f"✅ Import completed successfully")
            print(f"   Collections imported: {imported_collections}")
            print(f"   Total documents: {total_documents}")
            return 0
        else:
            print("❌ No collections were imported")
            return 1

    except Exception as e:
        print(f"❌ Error during import: {e}")
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
        'list': handle_list,
        'export': handle_export,
        'import': handle_import,
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
