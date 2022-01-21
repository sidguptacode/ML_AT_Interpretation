from __future__ import print_function
import argparse
import random
import json
import sys
sys.path.insert(0,'..')
from spreadsheet_editing.spreadsheet_editing import update_spreadsheet_labels

def main(args):
    f = open(args.labels_dict)
    labels_dict = json.load(f)
    update_spreadsheet_labels(labels_dict, token_path='../spreadsheet_editing/token.json', creds_path='../spreadsheet_editing/credentials.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Updates labels from labels_dict.json the spreadsheet.")
    parser.add_argument('--labels_dict', type=str, required=True)
    args = parser.parse_args()
    main(args)
