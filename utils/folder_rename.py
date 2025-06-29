import os
import re

from logger import add_file_handler, logger


def is_project_id(folder_name):
    """
    Check if a folder name matches the expected pattern (XX-XXXXX).

    Args:
        folder_name (str): Name of the folder to check

    Returns:
        bool: True if folder matches the pattern, False otherwise
    """
    number_pattern = re.compile(r"^\d{2}-\d{5}$")
    return bool(number_pattern.match(folder_name))


def get_scan_folders(data_dir):
    """
    Get the time-stamped folders inside the data directory.
    Only returns folders that match the pattern YYYYMMDD_HHMMSS.

    Args:
        data_dir (str): Path to the data directory

    Returns:
        list: List of folder names that match the timestamp pattern
    """
    time_pattern = re.compile(r"^\d{8}_\d{6}$")  # YYYYMMDD_HHMMSS pattern
    time_folders = []

    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and time_pattern.match(item):
            time_folders.append(item)

    return time_folders


def check_folder_contents(folder_path):
    """
    Check if a folder contains the required files.

    Args:
        folder_path (str): Path to the folder to check

    Returns:
        bool: True if folder contains both fuse_mesh.ply and fuse.ply, False otherwise
    """
    has_fuse_mesh = os.path.exists(os.path.join(folder_path, "fuse_mesh.ply"))
    has_fuse = os.path.exists(os.path.join(folder_path, "fuse.ply"))
    return has_fuse_mesh and has_fuse


def rename_scanned_folders(data_dir, folder_info):
    """
    Rename the time-stamped folders based on their contents.

    Args:
        data_dir (str): Path to the data directory
        folder_info (list): List of dictionaries containing folder information
    """
    # If one folder has the required files and the other doesn't, rename them
    if folder_info[0]["has_required_files"] != folder_info[1]["has_required_files"]:
        for folder in folder_info:
            prefix = "STD_" if folder["has_required_files"] else "ATR_"
            new_name = f"{prefix}{folder['name']}"
            new_path = os.path.join(data_dir, new_name)

            # Rename the folder
            if not os.path.exists(new_path):
                logger.info(f"  Renaming {folder['name']} to {new_name}")
                os.rename(folder["path"], new_path)
            else:
                logger.warning(
                    f"  Cannot rename {folder['name']} to {new_name}: destination already exists"
                )
    else:
        logger.info(
            f"  Folders in {data_dir} have same status for required files, skipping..."
        )


def process_project_folder(folder_path):
    """
    Process a single numbered folder to find and rename time-stamped folders.

    Args:
        folder_path (str): Path to the numbered folder
    """
    project_id = os.path.basename(folder_path)
    logger.info(f"Processing project folder: {project_id}")

    data_dir = os.path.join(folder_path, "data")
    if not os.path.isdir(data_dir):
        logger.warning(f"No 'data' folder found in {folder_path}, skipping...")
        return

    # Get time-stamped folders within data folder
    scan_folders = get_scan_folders(data_dir)

    # Check if we have exactly 2 folders
    if len(scan_folders) != 2:
        logger.warning(
            f"  Expected 2 scan folders in {project_id}/data, found {len(scan_folders)}, skipping..."
        )
        return

    # Check which folder contains the required files
    folder_info = [
        {
            "name": scan_folder,
            "path": os.path.join(data_dir, scan_folder),
            "has_required_files": check_folder_contents(
                os.path.join(data_dir, scan_folder)
            ),
        }
        for scan_folder in scan_folders
    ]

    # Rename the folders if needed
    rename_scanned_folders(data_dir, folder_info)


def update_revo_file(folder_path):
    """
    Update the .revo file content based on renamed folders in the data directory.
    If the correctly named .revo file doesn't exist, it tries to find and rename one.
    """
    project_id = os.path.basename(folder_path)
    revo_path = os.path.join(folder_path, f"{project_id}.revo")

    if not os.path.exists(revo_path):
        # If the specific .revo file doesn't exist, search for any .revo file
        try:
            found_revos = [f for f in os.listdir(folder_path) if f.endswith(".revo")]
        except FileNotFoundError:
            logger.error(
                f"  Could not find project folder at {folder_path} to search for .revo file."
            )
            return

        if len(found_revos) == 0:
            logger.warning(f"  No .revo file found in {project_id}, skipping update.")
            return
        elif len(found_revos) > 1:
            logger.warning(
                f"  Multiple .revo files found in {project_id}: {found_revos}. Skipping due to ambiguity."
            )
            return
        else:
            # Exactly one .revo file found, rename it.
            old_revo_path = os.path.join(folder_path, found_revos[0])
            logger.info(
                f"  Found '{found_revos[0]}', renaming to '{os.path.basename(revo_path)}'"
            )
            try:
                os.rename(old_revo_path, revo_path)
            except OSError as e:
                logger.error(f"  Failed to rename .revo file: {e}")
                return

    data_dir = os.path.join(folder_path, "data")
    sub_folder_names = os.listdir(data_dir)

    with open(revo_path, "r", encoding="utf-8") as f:
        content = f.read()
        original_content = content

    for sub_folder_name in sub_folder_names:
        if sub_folder_name.startswith("STD_") or sub_folder_name.startswith("ATR_"):
            old_name = sub_folder_name[4:]
            content = content.replace(f'"{old_name}"', f'"{sub_folder_name}"')

    if content != original_content:
        with open(revo_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.success(f"{project_id}.revo file updated successfully.")


def rename_folders(root_dir):
    """
    Traverse through the root directory to find and process project folders.

    Args:
        root_dir (str): Path to the root directory containing project folders
    """

    # Get all folders in the root directory
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)

        if os.path.isfile(subdir_path):
            continue

        logger.info(f"Processing subdirectory: {subdir_path}")
        for project_id in os.listdir(subdir_path):
            project_path = os.path.join(subdir_path, project_id)

            # Check if it's a directory and matches the pattern
            if os.path.isdir(project_path) and is_project_id(project_id):
                process_project_folder(project_path)
                update_revo_file(project_path)


if __name__ == "__main__":
    add_file_handler()
    root_dir = "/mnt/c/Users/10178/The Chinese University of Hong Kong/Wai Ping Fiona Yu (ORT) - 3D Back Images"

    rename_folders(root_dir)
