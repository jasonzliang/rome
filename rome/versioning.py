import os
import json
import datetime
from typing import List, Dict, Any, Optional
from ..logger import get_logger

def save_version(file_path: str, content: str,
                 changes: Optional[List[Dict[str, str]]] = None,
                 explanation: Optional[str] = None) -> int:
    """
    Save a versioned snapshot of a file with incremental version numbers.
    Version files are stored in a directory with the same name as the file
    but with ".ver" appended.

    Args:
        file_path: Path to the original file being versioned
        content: Current content of the file to save
        changes: Optional list of changes made in this version
        explanation: Optional explanation of changes made

    Returns:
        The version number assigned to this save
    """
    # Get the logger
    logger = get_logger()

    logger.info(f"Saving version for file: {file_path}")

    # Create versions directory with the .ver suffix
    base_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    file_base, file_ext = os.path.splitext(file_name)

    # Create the versions directory - same as file_path with .ver appended
    versions_dir = f"{file_path}.ver"

    logger.debug(f"Creating versions directory: {versions_dir}")
    os.makedirs(versions_dir, exist_ok=True)

    # Get the next version number by finding the highest existing version
    version_number = 1
    existing_versions = []

    # List all version files for this specific file
    try:
        for f in os.listdir(versions_dir):
            if f.startswith(file_base) and f.endswith(file_ext) and '_v' in f:
                try:
                    # Extract version number from filename
                    v_num = int(f.split('_v')[1].split('.')[0])
                    existing_versions.append(v_num)
                except (ValueError, IndexError):
                    logger.error(f"Skipping malformed version file: {f}")
                    continue

        # Set the new version number as one higher than the max existing version
        if existing_versions:
            version_number = max(existing_versions) + 1
            logger.debug(f"Found existing versions. New version will be: {version_number}")
        else:
            logger.debug("No existing versions found. Starting with version 1.")
    except Exception as e:
        logger.error(f"Error determining version number: {str(e)}")
        # Continue with version 1 as fallback

    # Create version file paths
    version_file_name = f"{file_base}_v{version_number}{file_ext}"
    version_file_path = os.path.join(versions_dir, version_file_name)

    meta_file_name = f"{file_base}_v{version_number}.meta.json"
    meta_file_path = os.path.join(versions_dir, meta_file_name)

    logger.debug(f"Version file will be saved as: {version_file_path}")
    logger.debug(f"Metadata file will be saved as: {meta_file_path}")

    # Create metadata
    metadata = {
        'version': version_number,
        'file_path': file_path,
        'timestamp': datetime.datetime.now().isoformat(),
        'changes': changes or [],
        'explanation': explanation or "No explanation provided"
    }

    # Save the versioned file
    try:
        with open(version_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Saved version {version_number} content to {version_file_path}")
    except Exception as e:
        logger.error(f"Failed to save version file: {str(e)}")
        raise

    # Save the metadata file
    try:
        with open(meta_file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Saved version {version_number} metadata to {meta_file_path}")
    except Exception as e:
        logger.error(f"Failed to save metadata file: {str(e)}")
        raise

    num_changes = len(changes) if changes else 0
    has_explanation = explanation is not None and explanation.strip() != ""

    if has_explanation:
        logger.info(f"Successfully saved version {version_number} with {num_changes} documented changes and explanation")
    else:
        logger.info(f"Successfully saved version {version_number} with {num_changes} documented changes")

    return version_number
