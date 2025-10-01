#!/usr/bin/env python
"""
ChromaDB Server Command Line Interface
Usage:
    python kb_server_cli.py start [--host HOST] [--port PORT] [--path PATH] [--force]
    python kb_server_cli.py stop [--force]
    python kb_server_cli.py restart [--host HOST] [--port PORT] [--path PATH]
    python kb_server_cli.py status
    python kb_server_cli.py delete [--collection NAME] [--metadata KEY:VALUE] [--force]
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
from typing import Dict, Optional, List, Any, Callable, Tuple
from datetime import datetime
from functools import wraps

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rome.kb_server import ChromaServerManager
from rome.logger import get_logger
from rome.config import LONG_SUMMARY_LEN
from rome.kb_client import EMBEDDING_MODELS

try:
    import chromadb
    from chromadb.utils.embedding_functions import (
       SentenceTransformerEmbeddingFunction, OpenAIEmbeddingFunction)
except ImportError as e:
    print(f"Import error: {e}")
    print("Install with: pip install chromadb")
    sys.exit(1)

logger = get_logger()
logger.configure({
    "level": "DEBUG",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "console": True,
    "include_caller_info": "rome"
})

DEFAULT_CONFIG = {'host': 'localhost', 'port': 8000, 'persist_path': None}
SERVER_CONFIG = DEFAULT_CONFIG.copy()

def compact_json_dump(data, file, indent=2):
    """Custom JSON dump that keeps embedding lists on single lines"""
    def format_item(obj, level=0):
        ind = ' ' * (indent * level)
        next_ind = ' ' * (indent * (level + 1))

        if isinstance(obj, dict):
            if not obj:
                return '{}'
            items = []
            for k, v in obj.items():
                # Special handling for 'embedding' key - keep list compact
                if k == 'embedding' and isinstance(v, list):
                    items.append(f'{next_ind}"{k}": {json.dumps(v)}')
                else:
                    items.append(f'{next_ind}"{k}": {format_item(v, level + 1)}')
            return '{\n' + ',\n'.join(items) + f'\n{ind}}}'
        elif isinstance(obj, list):
            if not obj:
                return '[]'
            # Check if this is a list of simple items or complex objects
            if all(not isinstance(item, (dict, list)) for item in obj):
                return json.dumps(obj)
            items = [format_item(item, level + 1) for item in obj]
            return '[\n' + ',\n'.join(f'{next_ind}{item}' for item in items) + f'\n{ind}]'
        else:
            return json.dumps(obj, ensure_ascii=False)

    file.write(format_item(data))

def error_handler(func: Callable) -> Callable:
    """Decorator for consistent error handling across commands"""
    @wraps(func)
    def wrapper(args):
        try:
            return func(args)
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
            return 130
        except Exception as e:
            print(f"❌ Error in {func.__name__}: {e}")
            traceback.print_exc()
            return 1
    return wrapper

def requires_server(func: Callable) -> Callable:
    """Decorator to ensure server is running before executing command"""
    @wraps(func)
    def wrapper(args):
        manager = get_server_manager()
        if not manager.is_running():
            print("❌ ChromaDB server is not running")
            print("   Use 'python kb_server_cli.py start' to start the server first")
            return 1
        return func(args)
    return wrapper

def confirm_action(message: str) -> bool:
    """Ask for user confirmation"""
    try:
        response = input(f"{message} (Y/n): ").strip().lower()
        return response not in ['n', 'no']
    except KeyboardInterrupt:
        print("\nCancelled.")
        return False

def parse_metadata_filter(metadata_str: str) -> Tuple[str, Any]:
    """Parse metadata filter string in format 'key:value' with type conversion"""
    if ':' not in metadata_str:
        raise ValueError(f"Metadata filter must be in format 'key:value', got: {metadata_str}")

    parts = metadata_str.split(':', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid metadata filter format: {metadata_str}")

    key, value_str = parts[0].strip(), parts[1].strip()
    if not key:
        raise ValueError("Metadata key cannot be empty")

    value = _convert_value_type(value_str)
    return key, value

def _convert_value_type(value_str: str) -> Any:
    """Convert string value to ChromaDB-compatible type (str, int, float, bool only)"""
    if value_str in ('True', 'true'):
        return True
    elif value_str in ('False', 'false'):
        return False

    try:
        if '.' not in value_str and value_str.lstrip('-').isdigit():
            return int(value_str)
    except ValueError:
        pass

    try:
        if '.' in value_str:
            return float(value_str)
    except ValueError:
        pass

    return value_str

class ServerManager:
    """Unified server management interface"""

    def __init__(self, config: Dict = None):
        self._manager = None
        self._client = None

    def get_manager(self, detached: bool = False):
        """Get server manager instance"""
        if detached:
            return self._get_detached_manager()
        return ChromaServerManager.get_instance(SERVER_CONFIG)

    def _get_detached_manager(self):
        """Get detached server manager that doesn't auto-cleanup"""
        original_register = atexit.register
        atexit.register = lambda *args, **kwargs: None
        try:
            manager = ChromaServerManager.get_instance(SERVER_CONFIG)
            manager._shutdown_registered = True
            return manager
        finally:
            atexit.register = original_register

    def get_client(self):
        """Get ChromaDB client"""
        if not self._client:
            manager = self.get_manager()
            self._client = chromadb.HttpClient(host=manager.host, port=manager.port)
        return self._client

# Global server manager instance
server_mgr = ServerManager()
get_server_manager = lambda: server_mgr.get_manager()
get_client = lambda: server_mgr.get_client()

class CollectionManager:
    """Handles collection operations with batching and metadata filtering"""

    @staticmethod
    def delete_documents_by_metadata(collection, metadata_key: str, metadata_value: Any,
                                   batch_size: int = 1000) -> Tuple[int, int]:
        """
        Delete documents matching metadata filter

        Returns:
            Tuple of (deleted_count, total_checked)
        """
        total_count = collection.count()
        deleted_count = 0
        total_checked = 0

        if total_count == 0:
            return 0, 0

        print(f"   Scanning {total_count} documents for metadata {metadata_key}:{metadata_value}")

        for offset in range(0, total_count, batch_size):
            result = collection.get(
                limit=batch_size,
                offset=offset,
                include=['metadatas']
            )

            batch_ids = result.get('ids', [])
            batch_metadatas = result.get('metadatas', [])

            matching_ids = []
            for i, metadata in enumerate(batch_metadatas):
                total_checked += 1
                if metadata and metadata.get(metadata_key) == metadata_value:
                    matching_ids.append(batch_ids[i])

            if matching_ids:
                collection.delete(ids=matching_ids)
                deleted_count += len(matching_ids)
                print(f"   Deleted {len(matching_ids)} documents from batch (offset {offset})")

        return deleted_count, total_checked

    @staticmethod
    def export_to_dict(collection, include_embeddings: bool = False, batch_size: int = 1000) -> Dict[str, Any]:
        """Export collection to dictionary with batching"""
        total_count = collection.count()
        export_data = {'collection_name': collection.name, 'count': total_count, 'data': []}

        include_fields = ['documents', 'metadatas'] + (['embeddings'] if include_embeddings else [])

        for offset in range(0, total_count, batch_size):
            results = collection.get(limit=batch_size, offset=offset, include=include_fields)

            for i in range(len(results['ids'])):
                item = {
                    'id': results['ids'][i],
                    'document': results['documents'][i] if results['documents'] else None,
                    'metadata': results['metadatas'][i] if results['metadatas'] else None
                }
                if include_embeddings and results.get('embeddings') is not None:
                    embedding = results['embeddings'][i]
                    # Convert numpy array to list for JSON serialization
                    item['embedding'] = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
                export_data['data'].append(item)

        return export_data

    @staticmethod
    def import_from_dict(client, data: Dict[str, Any], target_name: str = None,
                        embedding_model: str = None, overwrite: bool = False,
                        batch_size: int = 1000) -> bool:
        """Import collection from dictionary with optional embedding model"""
        collection_name = target_name or data['collection_name']

        # Setup embedding function if model specified
        embedding_fn = None
        if embedding_model:
            expected_dim = EMBEDDING_MODELS.get(embedding_model)
            if not expected_dim:
                print(f"   Error: Invalid embedding model '{embedding_model}'")
                return False

            is_sentence_transformer = expected_dim == 384

            if is_sentence_transformer:
                embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
            else:
                if not os.getenv('OPENAI_API_KEY'):
                    print(f"   Error: OPENAI_API_KEY required for {embedding_model}")
                    return False
                embedding_fn = OpenAIEmbeddingFunction(
                    model_name=embedding_model, api_key=os.getenv('OPENAI_API_KEY'))

        # Handle existing collection
        try:
            existing_collections = [col.name for col in client.list_collections()]
            if collection_name in existing_collections:
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
            if embedding_fn:
                collection = client.create_collection(collection_name, embedding_function=embedding_fn)
                print(f"   Created collection '{collection_name}' with {embedding_model}")
            else:
                collection = client.create_collection(collection_name)
                print(f"   Created collection '{collection_name}'")

            total_items = len(data['data'])
            imported_count = 0

            for start_idx in range(0, total_items, batch_size):
                batch_data = data['data'][start_idx:start_idx + batch_size]
                batch_kwargs = CollectionManager._prepare_batch(batch_data)

                collection.add(**batch_kwargs)
                imported_count += len(batch_data)

                if total_items > batch_size:
                    print(f"   Imported {imported_count}/{total_items} documents...")

            print(f"   Successfully imported {imported_count} documents to '{collection_name}'")
            return True

        except Exception as e:
            print(f"   Error importing collection '{collection_name}': {e}")
            return False

    @staticmethod
    def _prepare_batch(batch_data: List[Dict]) -> Dict:
        """Prepare batch data for ChromaDB insertion"""
        batch_ids, batch_documents, batch_metadatas, batch_embeddings = [], [], [], []
        has_documents = has_metadatas = has_embeddings = False

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

        kwargs = {'ids': batch_ids}
        if has_documents: kwargs['documents'] = batch_documents
        if has_metadatas: kwargs['metadatas'] = batch_metadatas
        if has_embeddings: kwargs['embeddings'] = batch_embeddings

        return kwargs

class OutputFormatter:
    """Handles all output formatting consistently"""

    @staticmethod
    def print_status_and_info(status: Dict, db_info: Dict):
        """Print combined status and database information"""
        if status['running']:
            print(f"✅ ChromaDB server is RUNNING")
            for key in ['URL', 'PID', 'Active clients', 'Host', 'Port', 'Startup timeout', 'Shutdown timeout']:
                value = status.get(key.lower().replace(' ', '_'), status.get(key.lower()))
                if value is not None:
                    unit = 's' if 'timeout' in key.lower() else ''
                    print(f"   {key}: {value}{unit}")
        else:
            print(f"❌ ChromaDB server is NOT RUNNING")
            print(f"   Would run at: {status['url']}")

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
                    dimension_info = OutputFormatter._get_collection_dimensions(col.get('name'))
                    dim_str = f" | {dimension_info}d" if dimension_info else ""
                    print(f"     - {col['name']} ({col['count']} documents{dim_str})")
            else:
                print(f"   Collections: None")

    @staticmethod
    def _get_collection_dimensions(collection_name: str) -> str:
        """Get embedding dimensions for a collection"""
        try:
            client = get_client()
            collection = client.get_collection(collection_name)

            if collection.count() == 0:
                return "empty"

            result = collection.get(limit=1, include=["embeddings"])
            embeddings = result.get("embeddings")

            if embeddings is None:
                return "no embeddings"

            if hasattr(embeddings, 'tolist'):
                embeddings = embeddings.tolist()

            if len(embeddings) > 0 and embeddings[0] is not None:
                return str(len(embeddings[0]))
            else:
                return "unknown"

        except Exception as e:
            return "error"

    @staticmethod
    def print_documents(documents: Dict, collection_name: str = None, limit: int = None):
        """Print formatted document information"""
        title = f"Documents in collection '{collection_name}'" if collection_name else "All documents"
        print(f"📄 {title}:")

        total_docs = displayed_docs = 0

        for col_name, docs in documents.items():
            if collection_name and col_name != collection_name:
                continue

            doc_count = len(docs)
            total_docs += doc_count

            if not collection_name:
                dimension_info = OutputFormatter._get_collection_dimensions(col_name)
                dim_str = f" | {dimension_info}d" if dimension_info and dimension_info not in ["empty", "error"] else ""
                print(f"\n┌── Collection: {col_name} ({doc_count} documents{dim_str}) ──┐")

            for doc in docs:
                if limit and displayed_docs >= limit:
                    print(f"\n💡 Showing first {limit} of {total_docs} total documents")
                    return

                OutputFormatter._print_single_document(doc, displayed_docs + 1)
                displayed_docs += 1

        print(f"\n{'─' * 80}")
        if displayed_docs == 0:
            target = f"collection '{collection_name}'" if collection_name else "any collection"
            print(f"   No documents found in {target}")
        else:
            if collection_name:
                print(f"   Total: {displayed_docs} documents in '{collection_name}'")
            else:
                collections_shown = len([name for name in documents.keys() if not collection_name or name == collection_name])
                print(f"   Total: {displayed_docs} documents across {collections_shown} collections")

    @staticmethod
    def _print_single_document(doc: Dict, doc_num: int):
        """Print a single document with formatting"""
        doc_id = doc.get('id', 'Unknown')
        content = doc.get('document', '')
        metadata = doc.get('metadata', {})

        maxlen = LONG_SUMMARY_LEN
        if len(content) > maxlen:
            truncated = content[:maxlen]
            last_space = truncated.rfind(' ')
            if last_space > maxlen * 0.8:
                content_preview = truncated[:last_space] + "..."
            else:
                content_preview = truncated + "..."
        else:
            content_preview = content

        print(f"\n📋 Document #{doc_num}")
        print(f"   ID: {doc_id}")

        if content_preview:
            print(f"   Content:")
            for line in content_preview.split('\n'):
                print(f"     {line}")
        else:
            print(f"   Content: (empty)")

        if metadata:
            print(f"   Metadata:")
            for key, value in sorted(metadata.items()):
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value, indent=6)
                    print(f"     {key}: {value_str}")
                else:
                    print(f"     {key}: {value}")

def create_parser() -> argparse.ArgumentParser:
    """Create modular argument parser"""
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

    # Delete all data (--force skips confirmation)
    python kb_server_cli.py delete --force

    # Delete specific collection with metadata filtering
    python kb_server_cli.py delete --collection my_collection --metadata key:value

    # List up to 10 documents in a collection
    python kb_server_cli.py list --collection my_collection --limit 10

    # Export specific collection with embeddings to JSON file
    python kb_server_cli.py export --collection my_collection --include-embeddings --output my_export.json

    # Import from JSON file and overwrite existing collection
    python kb_server_cli.py import --file my_export.json --overwrite
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    def add_server_args(parser):
        parser.add_argument('--host', default=DEFAULT_CONFIG['host'], help='Server host')
        parser.add_argument('--port', type=int, default=DEFAULT_CONFIG['port'], help='Server port')
        parser.add_argument('--path', default=DEFAULT_CONFIG['persist_path'], help='Data persistence path')

    def add_collection_arg(parser):
        parser.add_argument('--collection', help='Target collection name')

    def add_force_arg(parser):
        parser.add_argument('--force', action='store_true', help='Force action without confirmation')

    def add_batch_arg(parser):
        parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for operations')

    def add_metadata_arg(parser):
        parser.add_argument('--metadata', help='Filter by metadata key:value (e.g., source:web)')

    commands = [
        ('start', 'Start ChromaDB server', [add_server_args, add_force_arg]),
        ('stop', 'Stop ChromaDB server', [add_force_arg]),
        ('restart', 'Restart ChromaDB server', [add_server_args]),
        ('status', 'Show server status', []),
        ('delete', 'Delete database, collection, or filtered documents',
         [add_collection_arg, add_metadata_arg, add_force_arg]),
        ('list', 'List documents', [add_collection_arg, lambda p: p.add_argument('--limit', type=int, help='Limit results')]),
        ('export', 'Export to JSON', [add_collection_arg, add_batch_arg,
                                    lambda p: p.add_argument('--output', help='Output file'),
                                    lambda p: p.add_argument('--include-embeddings', action='store_true')]),
        ('import', 'Import from JSON', [add_collection_arg, add_batch_arg,
                                      lambda p: p.add_argument('--file', required=True, help='Input file'),
                                      lambda p: p.add_argument('--overwrite', action='store_true')])
    ]

    for cmd, help_text, arg_funcs in commands:
        cmd_parser = subparsers.add_parser(cmd, help=help_text)
        for arg_func in arg_funcs:
            arg_func(cmd_parser)

    return parser

@error_handler
def handle_start(args):
    """Handle start command"""
    manager = server_mgr.get_manager(detached=True)

    if manager.is_running() and not args.force:
        print("✅ ChromaDB server is already running")
        OutputFormatter.print_status_and_info(manager.get_status(), manager.get_database_info())
        return 0

    print("🚀 Starting ChromaDB server...")
    if args.force:
        print("   Force restart requested")

    success = manager.start(force_restart=args.force)

    if success:
        print("✅ ChromaDB server started successfully")
        OutputFormatter.print_status_and_info(manager.get_status(), manager.get_database_info())
        print("🔗 Server running in background. Use 'python kb_server_cli.py stop' to stop.")
        return 0
    else:
        print("❌ Failed to start ChromaDB server")
        print(f"   Try manually: chroma run --host {manager.host} --port {manager.port} --path {manager.persist_path}")
        return 1

@error_handler
def handle_stop(args):
    """Handle stop command"""
    manager = get_server_manager()

    if not manager.is_running():
        print("ℹ️  ChromaDB server is not running")
        return 0

    print("🛑 Stopping ChromaDB server...")
    if args.force:
        print("   Force stop requested")

    success = manager.stop(force=args.force)
    return 0 if success else 1

@error_handler
def handle_restart(args):
    """Handle restart command"""
    manager = get_server_manager()
    print("🔄 Restarting ChromaDB server...")

    success = manager.restart()
    if success:
        print("✅ ChromaDB server restarted successfully")
        OutputFormatter.print_status_and_info(manager.get_status(), manager.get_database_info())

    return 0 if success else 1

@error_handler
def handle_status(args):
    """Handle status command"""
    manager = get_server_manager()
    OutputFormatter.print_status_and_info(manager.get_status(), manager.get_database_info())
    return 0

@error_handler
def handle_delete(args):
    """Enhanced delete command with metadata filtering support"""
    manager = get_server_manager()

    if args.metadata and not args.collection:
        print("❌ --metadata filter requires --collection to be specified")
        return 1

    metadata_key = metadata_value = None
    if args.metadata:
        try:
            metadata_key, metadata_value = parse_metadata_filter(args.metadata)
        except ValueError as e:
            print(f"❌ Invalid metadata filter: {e}")
            return 1

    if args.metadata:
        if not manager.is_running():
            print("❌ ChromaDB server is not running")
            return 1

        try:
            client = get_client()
            collection = client.get_collection(args.collection)

            if not args.force:
                message = f"Delete documents in '{args.collection}' where {metadata_key}={metadata_value}?"
                if not confirm_action(message):
                    return 0

            print(f"🗑️  Deleting documents with metadata {metadata_key}:{metadata_value}...")
            deleted_count, total_checked = CollectionManager.delete_documents_by_metadata(
                collection, metadata_key, metadata_value
            )

            if deleted_count > 0:
                print(f"✅ Deleted {deleted_count} documents (checked {total_checked} total)")
            else:
                print(f"ℹ️  No documents found matching {metadata_key}:{metadata_value} (checked {total_checked} total)")

            return 0

        except Exception as e:
            if "not found" in str(e).lower():
                print(f"❌ Collection '{args.collection}' not found")
            else:
                print(f"❌ Error deleting documents: {e}")
            return 1

    elif args.collection:
        if not args.force and not confirm_action(f"Delete collection '{args.collection}'?"):
            return 0
        print(f"🗑️  Deleting collection '{args.collection}'...")
        success = manager.delete_collection(args.collection)
        message = f"Collection '{args.collection}'"
    else:
        if not args.force and not confirm_action("Delete ALL database data? This cannot be undone!"):
            return 0
        print("🗑️  Deleting entire database...")
        success = manager.clear_database(force=True)
        message = "Database"

    if 'success' in locals():
        print(f"✅ {message} deleted successfully" if success else f"❌ Failed to delete {message.lower()}")
        return 0 if success else 1

@error_handler
@requires_server
def handle_list(args):
    """Handle list command"""
    client = get_client()
    collections = client.list_collections()

    if not collections:
        print("📄 No collections found in database")
        return 0

    documents = {}
    for collection in collections:
        if args.collection and collection.name != args.collection:
            continue

        try:
            result = collection.get(include=['documents', 'metadatas'])
            documents[collection.name] = [
                {
                    'id': result['ids'][i],
                    'document': result['documents'][i] if result['documents'] else '',
                    'metadata': result['metadatas'][i] if result['metadatas'] else {}
                }
                for i in range(len(result['ids']))
            ]
        except Exception as e:
            print(f"⚠️  Error reading collection '{collection.name}': {e}")

    if not documents:
        target = f"Collection '{args.collection}'" if args.collection else "No documents"
        print(f"📄 {target} not found or empty")
        return 0

    OutputFormatter.print_documents(documents, args.collection, args.limit)
    return 0

@error_handler
@requires_server
def handle_export(args):
    """Handle export command with embedding model info"""
    client = get_client()
    collections = client.list_collections()

    if not collections:
        print("📄 No collections found in database")
        return 0

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = args.collection or "all"
    output_file = args.output or f"chroma_export_{suffix}_{timestamp}.json"

    print(f"📤 Exporting to '{output_file}'...")
    if args.include_embeddings:
        print("   Including embeddings in export")

    export_data = {
        'export_type': 'single_collection' if args.collection else 'all_collections',
        'collections': []
    }

    target_collections = [col for col in collections if not args.collection or col.name == args.collection]

    if args.collection and not target_collections:
        print(f"❌ Collection '{args.collection}' not found")
        return 1

    for collection in target_collections:
        print(f"   Exporting collection '{collection.name}'...")
        collection_data = CollectionManager.export_to_dict(
            collection, args.include_embeddings, args.batch_size
        )

        dimension = OutputFormatter._get_collection_dimensions(collection.name)
        embedding_model = None
        if dimension.isdigit():
            dim = int(dimension)
            embedding_model = next((model for model, d in EMBEDDING_MODELS.items() if d == dim), None)

        collection_data['embedding_model'] = embedding_model
        export_data['collections'].append(collection_data)

    with open(output_file, 'w', encoding='utf-8') as f:
        compact_json_dump(export_data, f, indent=2)

    total_collections = len(export_data['collections'])
    total_documents = sum(col['count'] for col in export_data['collections'])

    print(f"✅ Export completed successfully")
    print(f"   File: {output_file}")
    print(f"   Collections: {total_collections}")
    print(f"   Total documents: {total_documents}")
    print(f"   Embeddings included: {'Yes' if args.include_embeddings else 'No'}")

    return 0

@error_handler
@requires_server
def handle_import(args):
    """Handle import command with embedding model validation"""
    if not os.path.exists(args.file):
        print(f"❌ Input file '{args.file}' not found")
        return 1

    print(f"📥 Importing from '{args.file}'...")

    with open(args.file, 'r', encoding='utf-8') as f:
        import_data = json.load(f)

    if 'collections' not in import_data:
        print("❌ Invalid import file format: missing 'collections' key")
        return 1

    client = get_client()
    imported_collections = total_documents = 0

    for collection_data in import_data['collections']:
        if 'collection_name' not in collection_data:
            print("⚠️  Skipping collection with missing name")
            continue

        collection_name = args.collection or collection_data['collection_name']
        embedding_model = collection_data.get('embedding_model')

        if embedding_model:
            print(f"   Using embedding model: {embedding_model}")
            expected_dim = EMBEDDING_MODELS.get(embedding_model)
            if not expected_dim:
                print(f"   ⚠️  Unknown embedding model: {embedding_model}")

        print(f"   Importing collection '{collection_name}'...")

        if embedding_model and embedding_model in EMBEDDING_MODELS:
            success = CollectionManager.import_from_dict(
                client, collection_data, collection_name, embedding_model,
                args.overwrite, args.batch_size
            )
        else:
            success = CollectionManager.import_from_dict(
                client, collection_data, collection_name, None,
                args.overwrite, args.batch_size
            )

        if success:
            imported_collections += 1
            total_documents += collection_data.get('count', 0)

    if imported_collections > 0:
        print(f"✅ Import completed successfully")
        print(f"   Collections imported: {imported_collections}")
        print(f"   Total documents: {total_documents}")
        return 0
    else:
        print("❌ No collections were imported")
        return 1

def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    global SERVER_CONFIG
    for key in ['host', 'port', 'path']:
        if hasattr(args, key) and getattr(args, key) is not None:
            SERVER_CONFIG[key if key != 'path' else 'persist_path'] = getattr(args, key)

    commands = {
        'start': handle_start, 'stop': handle_stop, 'restart': handle_restart,
        'status': handle_status, 'delete': handle_delete, 'list': handle_list,
        'export': handle_export, 'import': handle_import
    }

    handler = commands.get(args.command)
    if not handler:
        print(f"❌ Unknown command: {args.command}")
        parser.print_help()
        return 1

    return handler(args)

if __name__ == "__main__":
    sys.exit(main())