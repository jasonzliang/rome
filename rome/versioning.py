import os
import json
import datetime
import hashlib
from typing import List, Dict, Any, Optional
from .logger import get_logger

META_DIR_EXT = 'rome'

def save_original(file_path: str, content: str) -> int:
    """
    Save the original unedited file into the meta directory.
    This is a convenience function that calls save_version with
    an appropriate explanation.

    Args:
        file_path: Path to the original file being versioned
        content: Content of the original file to save

    Returns:
        The version number assigned to the original file (typically 1)
    """
    logger = get_logger()
    logger.info(f"Saving original version for file: {file_path}")

    # If meta dir already exists you can assume original has already been saved
    if os.path.exists(f"{file_path}.{META_DIR_EXT}"):
        return

    return save_version(
        file_path=file_path,
        content=content,
        changes=[{"type": "initial", "description": "Original file content"}],
        explanation="Initial version: Original unedited file"
    )

def save_version(file_path: str, content: str,
                 changes: Optional[List[Dict[str, str]]] = None,
                 explanation: Optional[str] = None) -> int:
    """
    Save a versioned snapshot of a file with incremental version numbers.
    Version files are stored in a directory with the same name as the file
    but with meta dir ext appended. All metadata is stored in a index.json file.

    Args:
        file_path: Path to the original file being versioned
        content: Current content of the file to save
        changes: Optional list of changes made in this version
        explanation: Optional explanation of changes made

    Returns:
        The version number assigned to this save, or existing version if content hasn't changed
    """
    # Get the logger
    logger = get_logger()

    logger.info(f"Saving version for file: {file_path}")

    # Create file hash to check for duplicates
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    logger.debug(f"File content hash: {content_hash}")

    # Create versions directory with the meta dir ext
    base_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    file_base, file_ext = os.path.splitext(file_name)

    # Create the versions directory - same as file_path with meta dir ext appended
    versions_dir = f"{file_path}.{META_DIR_EXT}"

    if not os.path.exists(versions_dir):
        # logger.debug(f"Creating metadata directory: {versions_dir}")
        os.makedirs(versions_dir)

    # Path to the index file that contains all metadata
    index_file_path = os.path.join(versions_dir, "index.json")

    # Check if the index file exists and load it, otherwise create a new one
    if os.path.exists(index_file_path):
        try:
            with open(index_file_path, 'r', encoding='utf-8') as f:
                index = json.load(f)
                # logger.debug(f"Loaded existing index with {len(index.get('versions', []))} versions")
        except Exception as e:
            logger.error(f"Error loading index file: {str(e)}")
            index = {'versions': []}
    else:
        logger.debug("No existing index file. Creating new index.")
        index = {'versions': []}

    # Check if we already have a version with the same hash
    for version in index.get('versions', []):
        if version.get('hash') == content_hash:
            existing_version = version.get('version')
            logger.error(f"Content already exists in version {existing_version}. Skipping save.")
            return existing_version

    # Get the next version number
    if index.get('versions'):
        version_number = max(v.get('version', 0) for v in index['versions']) + 1
        # logger.debug(f"Found existing versions. New version will be: {version_number}")
    else:
        version_number = 1
        # logger.debug("No existing versions found. Starting with version 1.")

    # Create version file path
    version_file_name = f"{file_base}_v{version_number}{file_ext}"
    version_file_path = os.path.join(versions_dir, version_file_name)

    logger.debug(f"Version file will be saved as: {version_file_path}")

    # Create metadata entry
    timestamp = datetime.datetime.now().isoformat()
    metadata = {
        'version': version_number,
        'file_path': file_path,
        'timestamp': timestamp,
        'hash': content_hash,
        'changes': changes or [],
        'explanation': explanation or "No explanation provided"
    }

    # Save the versioned file
    try:
        with open(version_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        # logger.info(f"Saved version {version_number} content to {version_file_path}")
    except Exception as e:
        logger.error(f"Failed to save version file: {str(e)}")
        raise

    # Add the metadata to the index and save it
    try:
        index['versions'].append(metadata)

        # Sort versions by version number
        index['versions'].sort(key=lambda x: x.get('version', 0))

        with open(index_file_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)
        # logger.info(f"Updated index file with version {version_number} metadata")
    except Exception as e:
        logger.error(f"Failed to update index file: {str(e)}")
        raise

    num_changes = len(changes) if changes else 0
    logger.info(f"Successfully saved version {version_number} with {num_changes} documented changes")

    return version_number
