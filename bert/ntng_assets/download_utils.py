import os
import shutil
import tarfile
import zipfile
from urllib.parse import urlparse

import requests
from tqdm import tqdm


def download_file(url, destination_folder):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    file_name = os.path.basename(urlparse(url).path)
    file_path = os.path.join(destination_folder, file_name)
    os.makedirs(destination_folder, exist_ok=True)

    with open(file_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")


def unzip_file(file_path, destination_folder):
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(destination_folder)


def untar_file(file_path, destination_folder):
    with tarfile.open(file_path, "r") as tar_ref:
        tar_ref.extractall(destination_folder)


def move_files(source_folder, destination_folder):
    files = os.listdir(source_folder)
    for file_name in files:
        shutil.move(os.path.join(source_folder, file_name), destination_folder)


def move_all_files_to_top_folder(top_folder, subfolder):
    """Moves all files within a deep subfolder to the top folder.

    Args:
      top_folder: The top folder.
      subfolder: The subfolder to move the files from.
    """

    for root, dirs, files in os.walk(subfolder):
        for file in files:
            shutil.move(os.path.join(root, file), top_folder)
