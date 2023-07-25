import os
from urllib.parse import urlparse

from download_utils import download_file, move_files, unzip_file


def main():
    # Parameters
    download_url = "https://www.dropbox.com/s/l2v3v2wgwuiwkxq/c4-subset_WordPiecex32768_e0501aeb87699de7500dc91a54939f44.zip?dl=1"
    download_destination_folder = unzip_destination_folder = "../outputs/data/"
    move_destination_folder = "./outputs/data/c4-subset_WordPiecex32768_e0501aeb87699de7500dc91a54939f44/"

    print("Downloading file...")
    download_file(download_url, download_destination_folder)

    print("Unzipping file...")
    zip_file_path = os.path.join(download_destination_folder, os.path.basename(urlparse(download_url).path))
    unzip_file(zip_file_path, unzip_destination_folder)

    print("Moving files...")
    move_files(unzip_destination_folder, move_destination_folder)
    print("Done.")


if __name__ == "__main__":
    main()
