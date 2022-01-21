import owncloud
import os
import tqdm
from tqdm import tqdm
from pathlib import Path
from update_spreadsheet import append_urls_to_spreadsheet
import argparse

"""
    Uploads individual well images to ownCloud.
    NOTE: This file is not used, since we upload to GitLab so annotation GET requests move faster.
    But perhaps it will be used when I handoff the code.
"""

def upload_npz_to_oc():
    last_path_dir = str(Path.cwd()).split('/')[-1]
    if last_path_dir != 'individual_well_imgs':
        raise Exception("Must be in the individual_well_imgs directory to use this script.")

    public_link = 'https://cloud.digiomics.com/index.php/s/1yMusMFezljkjXK'

    oc = owncloud.Client.from_public_link(public_link)

    oc_imglist = oc.list('.')
    oc_imglist_names = [oc_img.path.split('/')[1] for oc_img in oc_imglist]
    # curr_imglist_names = [curr_imgname for curr_imgname in os.listdir('.') if curr_imgname not in oc_imglist_names]
    curr_imglist_names_temp = [curr_imgname for curr_imgname in os.listdir('.') if curr_imgname in oc_imglist_names]

    # Begin uploading the npz files that haven't been uploaded.
    print("Uploading npz trays to ownCloud")
    for curr_imgname in tqdm(curr_imglist_names):
        oc.drop_file(curr_imgname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="For running different kinds of feature-extraction experiments on LAT trays.")
    parser.add_argument('--img_folder', type=str, required=True)
    args = parser.parse_args()
    get_gitlab_urls(args.img_folder)
    img_url_list = get_gitlab_urls()
    # img_url_list = upload_to_oc()
    append_urls_to_spreadsheet(img_url_list)
