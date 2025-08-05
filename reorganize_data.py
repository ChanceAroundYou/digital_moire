"""
Data reorganization module for digital moire analysis.

This module provides functionality to reorganize and copy files from a complex source
directory to a simplified flat structure organized by project ID.

FROM:
- source_dir
    - date_dir_1 (YYYYMMDD)
        - project_id_1 (XX-XXXXX)
            - data
                - STDD_xxx
                    - fuse.ply
                    - fuse_mesh.ply
                - STD_xxx
                    - fuse.ply
                    - fuse_mesh.ply
                - ATR_xxx
                    - fuse.ply
                    - fuse_mesh.ply
        - project_id_2
        - project_id_3
        ...
        - X-rays
            - project_id_1_LAT_xxx.jpg
            - project_id_1_PA_xxx.jpg
            ...
    - date_dir_2
    - date_dir_3
    ...
TO:
- dest_dir
    - project_id_1 (XX-XXXXX)
        - STDD_fuse.ply
        - STDD_fuse_mesh.ply
        - STD_fuse.ply
        - STD_fuse_mesh.ply
        - ATR_fuse.ply
        - ATR_fuse_mesh.ply
        - LAT_xxx.jpg
        - PA_xxx.jpg
    - project_id_2
    - project_id_3
    ...

USAGE:
    # Use default source and destination directories
    python reorganize_data.py

    # Specify only source directory (use default destination)
    python reorganize_data.py /path/to/source

    # Specify both source and destination directories
    python reorganize_data.py /path/to/source /path/to/destination

    # Show help information
    python reorganize_data.py --help

"""

import argparse
import hashlib
import os
import re
import shutil

from utils.logger import add_file_handler, logger


def is_date(folder_name):
    """
    Check if a folder name matches the expected pattern (YYYYMMDD).

    Args:
        folder_name (str): Name of the folder to check

    Returns:
        bool: True if folder matches the pattern, False otherwise
    """
    number_pattern = re.compile(r"^\d{8}$")
    if not number_pattern.match(folder_name):
        return False
    year = int(folder_name[:4])
    month = int(folder_name[4:6])
    day = int(folder_name[6:8])
    try:
        import datetime

        datetime.date(year, month, day)
        return True
    except ValueError:
        return False


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


def copy_scan_files(scan_dir, project_dest_dir, scan_type):
    """
    Copy scan files (STD or ATR) from source to the output directory.

    Args:
        scan_dir (str): Path to the scan folder (STD or ATR)
        project_dest_dir (str): Path to the project output directory
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
        dst_path = os.path.join(project_dest_dir, dst_name)
        if os.path.exists(src_path):
            copy_with_hash_check(src_path, dst_path)
        else:
            logger.debug(f"Source file {src_path} not found, skipping")


def copy_xray_files(xrays_dir, dest_dir):
    """
    Copy X-ray files for all projects from the X-rays folder.
    Scans the directory to find all X-ray files that match the project ID pattern.

    Args:
        xrays_dir (str): Path to the X-rays directory
        dest_dir (str): Path to the output directory
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
            project_dest_dir = os.path.join(dest_dir, project_id)
            os.makedirs(project_dest_dir, exist_ok=True)

            # Copy the X-ray file
            src_path = os.path.join(xrays_dir, xray_file)
            dst_path = os.path.join(project_dest_dir, f"{xray_type}_{suffix}.jpg")
            copy_with_hash_check(src_path, dst_path)


def process_project(date_dir_path, project_id, dest_dir):
    """
    Process a single project folder, copying all relevant files.

    Args:
        subdir_path (str): Path to the parent directory containing the project
        project_id (str): Project ID
        dest_dir (str): Base output directory
    """
    project_path = os.path.join(date_dir_path, project_id)
    if not os.path.isdir(project_path):
        logger.warning(f"Project path {project_path} is not a directory")

    logger.info(f"Processing project folder: {project_id}")

    # Create project output directory
    project_dest_dir = os.path.join(dest_dir, project_id)
    os.makedirs(project_dest_dir, exist_ok=True)

    # Process data files
    project_data_dir = os.path.join(project_path, "data")

    # Skip if data directory doesn't exist
    if not os.path.isdir(project_data_dir):
        logger.warning(f"No data directory found in project {project_id}")

    # Find STD, STDD and ATR folders directly
    for scan_name in os.listdir(project_data_dir):
        scan_dir_path = os.path.join(project_data_dir, scan_name)

        # Skip non-directory items
        if not os.path.isdir(scan_dir_path):
            continue

        if scan_name.startswith("STDD_"):
            logger.info(f"Found STDD scan folder: {scan_name}")
            copy_scan_files(scan_dir_path, project_dest_dir, "STDD")
        elif scan_name.startswith("STD_"):
            logger.info(f"Found STD scan folder: {scan_name}")
            copy_scan_files(scan_dir_path, project_dest_dir, "STD")
        elif scan_name.startswith("ATR_"):
            logger.info(f"Found ATR scan folder: {scan_name}")
            copy_scan_files(scan_dir_path, project_dest_dir, "ATR")


def reorganize_data(source_dir, dest_dir):
    """
    Copy all relevant files from the source directory to the destination directory.

    Args:
        source_dir (str): Root directory containing all the project folders
        dest_dir (str): Output directory where files will be copied to
    """
    # Ensure output directory exists
    os.makedirs(dest_dir, exist_ok=True)
    logger.info(f"Starting data rearrangement from {source_dir} to {dest_dir}")

    # Process all date folders
    for date_dir in os.listdir(source_dir):
        if not is_date(date_dir):
            continue

        date_dir_path = os.path.join(source_dir, date_dir)

        if not os.path.isdir(date_dir_path):
            continue

        logger.info(f"Processing date folder: {date_dir}")

        for project_id in os.listdir(date_dir_path):
            if not is_project_id(project_id):
                continue

            process_project(date_dir_path, project_id, dest_dir)
            logger.success(f"Completed processing project: {project_id}")

        # Find and process X-rays directory
        xrays_dir = os.path.join(date_dir_path, "X-rays")
        if os.path.isdir(xrays_dir):
            logger.info(f"Processing X-rays directory in {date_dir}...")
            copy_xray_files(xrays_dir, dest_dir)
            logger.success(f"Completed processing X-rays directory in {date_dir}...")


if __name__ == "__main__":
    add_file_handler()
    parser = argparse.ArgumentParser(
        description="Reorganize digital moire data from complex nested structure to flat project-based structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        "   python reorganize_data.py                                    # Use default paths\n"
        "   python reorganize_data.py /path/to/source                   # Specify source only\n"
        "   python reorganize_data.py /path/to/source /path/to/dest     # Specify both paths\n"
        "   python reorganize_data.py --help                            # Show this help message\n\n"
        "The script reorganizes data from a complex nested directory structure to a simplified\n"
        "flat structure organized by project ID. It copies scan files (STD, STDD, ATR) and\n"
        "X-ray images while maintaining file integrity through hash checking.",
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=str,
        default="/mnt/c/Users/10178/OneDrive - The Chinese University of Hong Kong/Wai Ping Fiona Yu (ORT)'s files - 3D Back Images",
        help="Source directory containing date folders with project data (default: %(default)s)",
    )
    parser.add_argument(
        "dest",
        nargs="?",
        type=str,
        default="data",
        help="Destination directory for reorganized data (default: %(default)s)",
    )
    args = parser.parse_args()

    source_dir = args.source
    dest_dir = args.dest

    logger.info(f"Source directory: {source_dir}")
    logger.info(f"Destination directory: {dest_dir}")

    reorganize_data(source_dir, dest_dir)
    logger.success("Data rearrangement complete.")
