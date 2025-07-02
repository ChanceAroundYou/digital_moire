import hashlib
import os
import re
import shutil

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


def get_file_hash(file_path):
    """
    Calculate the MD5 hash of a file.

    Args:
        file_path (str): Path to the file

    Returns:
        str: MD5 hash of the file
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def copy_with_hash_check(src_path, dst_path):
    """
    Copy a file from source to destination with hash checking.
    If the destination file already exists, check the hash:
    - If hashes match, don't copy
    - If hashes don't match, keep the newer file

    Args:
        src_path (str): Source file path
        dst_path (str): Destination file path
    """
    # Create destination directory if it doesn't exist
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # If destination doesn't exist, just copy
    if not os.path.exists(dst_path):
        logger.info(f"Copying {src_path} to {dst_path}")
        shutil.copy2(src_path, dst_path)
        return

    # Compare hashes
    src_hash = get_file_hash(src_path)
    dst_hash = get_file_hash(dst_path)

    if src_hash == dst_hash:
        logger.debug(f"File {dst_path} already exists with identical hash, skipping")
    else:
        # Compare modification times
        src_mtime = os.path.getmtime(src_path)
        dst_mtime = os.path.getmtime(dst_path)

        if src_mtime > dst_mtime:
            logger.success(f"Copying {src_path} to {dst_path}")
            shutil.copy2(src_path, dst_path)
        else:
            logger.debug(f"Destination file {dst_path} is newer, keeping it")


def copy_scan_files(scan_dir, project_output_dir, scan_type):
    """
    Copy scan files (STD or ATR) from source to the output directory.

    Args:
        scan_dir (str): Path to the scan folder (STD or ATR)
        project_output_dir (str): Path to the project output directory
        scan_type (str): Type of scan ('STD' or 'ATR')
    """
    # Define mapping of source files to destination names
    file_mappings = {
        "fuse.ply": f"{scan_type}_fuse.ply",
        "fuse_mesh.ply": f"{scan_type}_fuse_mesh.ply",
    }

    # Copy each file if it exists
    for src_name, dst_name in file_mappings.items():
        src_path = os.path.join(scan_dir, src_name)
        dst_path = os.path.join(project_output_dir, dst_name)
        if os.path.exists(src_path):
            copy_with_hash_check(src_path, dst_path)
        else:
            logger.debug(f"Source file {src_path} not found, skipping")


def copy_xray_files(xrays_dir, output_dir):
    """
    Copy X-ray files for all projects from the X-rays folder.
    Scans the directory to find all X-ray files that match the project ID pattern.

    Args:
        xrays_dir (str): Path to the X-rays directory
        output_dir (str): Path to the output directory
    """
    if not os.path.isdir(xrays_dir):
        logger.warning(f"X-rays directory {xrays_dir} not found")
        return

    # Pattern to match X-ray filenames: project_id, type (LAT/PA), and suffix
    xray_pattern = re.compile(r"^(\d{2}-\d{5})_(LAT|PA)_(.*)\.jpg$")

    for xray_file in os.listdir(xrays_dir):
        match = xray_pattern.match(xray_file)
        if match:
            project_id = match.group(1)
            xray_type = match.group(2)  # LAT or PA
            suffix = match.group(3)

            # Create project output directory
            project_output_dir = os.path.join(output_dir, project_id)
            os.makedirs(project_output_dir, exist_ok=True)

            # Copy the X-ray file
            src_path = os.path.join(xrays_dir, xray_file)
            dst_path = os.path.join(project_output_dir, f"{xray_type}_{suffix}.jpg")
            copy_with_hash_check(src_path, dst_path)


def process_project(subdir_path, project_id, output_dir):
    """
    Process a single project folder, copying all relevant files.

    Args:
        subdir_path (str): Path to the parent directory containing the project
        project_id (str): Project ID
        output_dir (str): Base output directory
    """
    project_path = os.path.join(subdir_path, project_id)
    if not os.path.isdir(project_path):
        logger.warning(f"Project path {project_path} is not a directory")

    logger.info(f"Processing project folder: {project_id}")

    # Create project output directory
    project_output_dir = os.path.join(output_dir, project_id)
    os.makedirs(project_output_dir, exist_ok=True)

    # Process data files
    project_data_dir = os.path.join(project_path, "data")

    # Skip if data directory doesn't exist
    if not os.path.isdir(project_data_dir):
        logger.warning(f"No data directory found in project {project_id}")

    # Find STD and ATR folders directly
    for scan_name in os.listdir(project_data_dir):
        scan_dir_path = os.path.join(project_data_dir, scan_name)

        # Skip non-directory items
        if not os.path.isdir(scan_dir_path):
            continue

        # Process STD or ATR folder
        if scan_name.startswith("STD"):
            logger.info(f"Found STD scan folder: {scan_name}")
            copy_scan_files(scan_dir_path, project_output_dir, "STD")
        elif scan_name.startswith("ATR"):
            logger.info(f"Found ATR scan folder: {scan_name}")
            copy_scan_files(scan_dir_path, project_output_dir, "ATR")


def data_rearrange(root_dir, output_dir):
    """
    Copy all relevant files from the project folders to the output directory.
    Also scans X-rays directory for files to copy.

    Args:
        root_dir (str): Root directory containing all the project folders
        output_dir (str): Output directory where files will be copied to
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Starting data rearrangement from {root_dir} to {output_dir}")

    # Process all project folders
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)

        if not os.path.isdir(subdir_path):
            continue

        logger.info(f"Processing subdirectory: {subdir_path}")

        for project_id in os.listdir(subdir_path):
            if not is_project_id(project_id):
                continue

            process_project(subdir_path, project_id, output_dir)
            logger.success(f"Completed processing project: {project_id}")

        # Find and process X-rays directory
        xrays_dir = os.path.join(subdir_path, "X-rays")
        if os.path.isdir(xrays_dir):
            logger.info(f"Processing X-rays directory in {subdir}...")
            copy_xray_files(xrays_dir, output_dir)
            logger.success(f"Completed processing X-rays directory in {subdir}...")


if __name__ == "__main__":
    add_file_handler()
    root_dir = "/mnt/c/Users/10178/The Chinese University of Hong Kong/Wai Ping Fiona Yu (ORT) - 3D Back Images"
    output_dir = "data"

    data_rearrange(root_dir, output_dir)
    logger.success("Data rearrangement complete.")
