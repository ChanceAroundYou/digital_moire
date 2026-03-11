import hashlib
import os
import re
import shutil

from ..logger import logger


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