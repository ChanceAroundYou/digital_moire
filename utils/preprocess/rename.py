import json
import os

# from .data.folder import is_date, is_project_id
from ..logger import logger


def find_revo_file(folder_path, project_id):
    """
    Find and return the path to the .revo file for the project.
    If the correctly named .revo file doesn't exist, search for any .revo file and rename it.

    Args:
        folder_path (str): Path to the project folder
        project_id (str): Project ID

    Returns:
        str: Path to the .revo file, or None if not found/error
    """
    revo_path = os.path.join(folder_path, f"{project_id}.revo")

    if os.path.exists(revo_path):
        return revo_path

    # Search for any .revo file
    try:
        found_revos = [f for f in os.listdir(folder_path) if f.endswith(".revo")]
    except FileNotFoundError:
        logger.error(
            f"Could not find project folder at {folder_path} to search for .revo file."
        )
        return None

    if len(found_revos) == 0:
        logger.warning(f"No .revo file found in {project_id}, skipping update.")
        return None
    elif len(found_revos) > 1:
        logger.warning(
            f"Multiple .revo files found in {project_id}: {found_revos}. Skipping due to ambiguity."
        )
        return None
    else:
        # Exactly one .revo file found, rename it
        old_revo_path = os.path.join(folder_path, found_revos[0])
        logger.info(
            f"Found '{found_revos[0]}', renaming to '{os.path.basename(revo_path)}'"
        )
        try:
            os.rename(old_revo_path, revo_path)
            return revo_path
        except OSError as e:
            logger.error(f"Failed to rename .revo file: {e}")
            return None


def load_revo_json(revo_path, project_id):
    """
    Load and parse the JSON content from a .revo file.

    Args:
        revo_path (str): Path to the .revo file
        project_id (str): Project ID for logging

    Returns:
        dict: Parsed JSON data, or None if error
    """
    try:
        with open(revo_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON in {project_id}.revo: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to read {project_id}.revo: {e}")
        return None


def save_revo_json(revo_path, revo_data, project_id):
    """
    Save the updated JSON data back to the .revo file.

    Args:
        revo_path (str): Path to the .revo file
        revo_data (dict): JSON data to save
        project_id (str): Project ID for logging

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(revo_path, "w", encoding="utf-8") as f:
            json.dump(revo_data, f, indent=4, ensure_ascii=False)
        logger.success(f"{project_id}.revo file updated successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to write updated {project_id}.revo: {e}")
        return False


def rename_data_folder(data_dir, old_name, new_name):
    """
    Rename a folder in the data directory.

    Args:
        data_dir (str): Path to the data directory
        old_name (str): Current folder name
        new_name (str): New folder name

    Returns:
        bool: True if successful or not needed, False if failed
    """
    old_folder_path = os.path.join(data_dir, old_name)
    new_folder_path = os.path.join(data_dir, new_name)

    if not os.path.exists(old_folder_path):
        return True  # Nothing to rename

    if os.path.exists(new_folder_path):
        logger.warning(
            f"Target folder '{new_name}' already exists, skipping folder rename"
        )
        return True  # Target exists, but this is not an error

    logger.info(f"Renaming folder '{old_name}' to '{new_name}'")
    try:
        os.rename(old_folder_path, new_folder_path)
        return True
    except OSError as e:
        logger.error(f"Failed to rename folder: {e}")
        return False


def process_node(node, data_dir, project_id):
    """
    Process a single node from the .revo file.
    Check if name starts with STD_, STDD_, ATR_ and update guid/folder if needed.

    Args:
        node (dict): Node data from the .revo file
        data_dir (str): Path to the data directory
        project_id (str): Project ID for logging

    Returns:
        bool: True if node was updated, False otherwise
    """
    node_name = node.get("name", "")
    node_guid = node.get("guid", "")

    # Check if name starts with STD_, STDD_, or ATR_
    if not node_name.startswith(("STD_", "STDD_", "ATR_")):
        return False

    # Check if guid differs from name
    if node_guid == node_name:
        return False  # No update needed

    logger.info(
        f"Found mismatch in {project_id}: name='{node_name}', guid='{node_guid}'"
    )

    # Rename folder in data directory if it exists
    if not rename_data_folder(data_dir, node_guid, node_name):
        return False  # Failed to rename folder, skip guid update

    # Update guid in the node
    logger.info(f"Updating guid from '{node_guid}' to '{node_name}'")
    node["guid"] = node_name
    return True


def rename_folder(folder_path):
    """
    Update the .revo file content based on nodes information.
    If node name starts with STD_, STDD_, ATR_ and differs from guid,
    update the guid and rename corresponding data folder.
    """
    project_id = os.path.basename(folder_path)

    # Find the .revo file
    revo_path = find_revo_file(folder_path, project_id)
    if not revo_path:
        return

    # Load JSON data
    revo_data = load_revo_json(revo_path, project_id)
    if not revo_data:
        return

    # Check if data directory exists
    data_dir = os.path.join(folder_path, "data")
    if not os.path.exists(data_dir):
        logger.warning(f"No data directory found in {project_id}")
        return

    # Process each node
    updated = False
    for node in revo_data.get("nodes", []):
        if process_node(node, data_dir, project_id):
            updated = True

    # Save updated JSON if changes were made
    if updated:
        save_revo_json(revo_path, revo_data, project_id)
    else:
        logger.info(f"No updates needed for {project_id}.revo")
