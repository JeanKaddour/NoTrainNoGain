import argparse
import os
from urllib.parse import urlparse

from download_utils import download_file, move_all_files_to_top_folder, untar_file

URLS = {
    "minipile": "https://www.dropbox.com/s/cmvq41yx16f7gmj/minipile_losses.tar?dl=1",
    "bcwk": "https://www.dropbox.com/s/vt8ggnhscyb8ov1/bcwk_il_losses.tar?dl=0",
    "c4": "https://www.dropbox.com/s/a7qs3vkccyly3h1/c4_il_losses.tar?dl=0",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="minipile", choices=["minipile", "c4"], help="dataset to download")
    args = parser.parse_args()

    # Parameters
    download_url = URLS[args.dataset]
    download_destination_folder = f"./tmp/"
    move_destination_folder = f"./{args.dataset}/"
    os.makedirs(move_destination_folder, exist_ok=True)
    if os.path.exists(os.path.join(download_destination_folder, os.path.basename(urlparse(download_url).path))):
        print("File already exists, skipping download.")
    else:
        print("Downloading file...")
        download_file(download_url, download_destination_folder)

    print("Unzipping file...")
    tar_file_path = os.path.join(download_destination_folder, os.path.basename(urlparse(download_url).path))
    untar_file(tar_file_path, move_destination_folder)

    move_all_files_to_top_folder(move_destination_folder, os.path.join(move_destination_folder, "outputs"))

    print("Done.")


if __name__ == "__main__":
    main()
