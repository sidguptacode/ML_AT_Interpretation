from __future__ import print_function
import argparse
import random
import sys
sys.path.insert(0,'..')
from spreadsheet_editing.spreadsheet_editing import append_urls_to_spreadsheet

def main(args):
    if args.prefix == 'gitlab':
        prefix = 'https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/raw/master/indiv_well_imgs/'
    else:
        # Should have other options here
        raise Exception("Currently can only Gitlab URLs")

    f = open(args.files_uploaded, "r")
    # The [:-1] removes trailing '\n` characters
    imgs_uploaded = [img_file[:-1] for img_file in f.readlines()]
    img_url_list = [f"{prefix}{img_name}" for img_name in imgs_uploaded]
    random.shuffle(img_url_list)
    append_urls_to_spreadsheet(img_url_list, token_path='../spreadsheet_editing/token.json', creds_path='../spreadsheet_editing/credentials.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Adds URLs from files_uploaded.txt to the spreadsheet.")
    parser.add_argument('--files_uploaded', type=str, required=True)
    parser.add_argument('--prefix', type=str, required=True)
    args = parser.parse_args()
    main(args)
