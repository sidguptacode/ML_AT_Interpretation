import owncloud
import tqdm
from tqdm import tqdm
import os
from pathlib import Path
import argparse
import sys
sys.path.insert(0,'..')
from helpers import create_path
from constants import OWNCLOUD_PUBLIC_LINK


def download_from_oc(save_dirpath, num_trays_to_download=2, trays_to_download=None):
    create_path(save_dirpath, directory=True)

    if save_dirpath[-1] != os.sep:
        save_dirpath += os.sep
    save_dirpath += '.'

    oc = owncloud.Client.from_public_link(OWNCLOUD_PUBLIC_LINK)

    oc_dirlist = oc.list('.')
    curr_dirlist_names = os.listdir(save_dirpath)
    oc_dirlist_names = [oc_dir.path.split('/')[1] for oc_dir in oc_dirlist]
    oc_dirlist_names = [oc_dirname for oc_dirname in oc_dirlist_names if oc_dirname not in curr_dirlist_names]
    if trays_to_download is not None:
        oc_dirlist_names_temp = []
        for tray_to_download in trays_to_download:
            for oc_dirname in oc_dirlist_names:
                if tray_to_download in oc_dirname:
                    oc_dirlist_names_temp.append(oc_dirname)
        oc_dirlist_names = oc_dirlist_names_temp

    # We reverse so that we download the most recent trays first.
    oc_dirlist_names.reverse()
    for oc_dirname in oc_dirlist_names:
        if num_trays_to_download == 0:
            break
        img_list = oc.list(f"/{oc_dirname}/")
        imgname_list = [img.path.split('/')[2] for img in img_list]
        if 'meta.yml' not in imgname_list:
            print(f"Could not download {oc_dirname}, as it doesn't contain a .yml file.")
            continue
        new_dir = create_path(f"{save_dirpath}{os.sep}{oc_dirname}", directory=True)
        if new_dir:
            print(f"Downloading {oc_dirname}")
            for img in tqdm(img_list):
                imgname = img.path.split('/')[2]
                oc.get_file(img.path, f"{save_dirpath}{os.sep}{oc_dirname}{os.sep}{imgname}")
            num_trays_to_download -= 1


if __name__ == "__main__":
    """
        A Python script that allows you to quickly download trays from ownCloud.
        If you specify the argument `num_trays`, then it will download the `num_trays` most recent trays.
        If you specifiy `folder1` or `folder2`, it will download those two folders.
    """
    parser = argparse.ArgumentParser(description="Downloads images of LAT trays from ownCloud.")
    parser.add_argument('--save_dirpath', type=str, required=True)
    parser.add_argument('--num_trays', type=int, required=False)
    parser.add_argument('--folder1', type=str, required=False)
    parser.add_argument('--folder2', type=str, required=False)
    args = parser.parse_args()
    trays_to_download = [args.folder1, args.folder2]
    trays_to_download = [tray for tray in trays_to_download if tray is not None]
    if trays_to_download == []:
        trays_to_download = None

    num_trays = 2 if args.num_trays is None else args.num_trays
    download_from_oc(args.save_dirpath, num_trays, trays_to_download)