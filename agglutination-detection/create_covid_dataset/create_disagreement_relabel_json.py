import pandas as pd
import json

"""
    Usage: Paste the URLs from high_disagreements.txt in a seperate google sheets, update
    the website to annotate that sheet, export that sheet as a csv, and insert it here
    to be converted to JSON, so we can update the whole spreadsheet with the checked labels.
"""

df = pd.read_csv('./relabel.csv')
urls = df['urls'].values
labels = df['labels'].values

json_out = {urls[i]: int(labels[i]) for i in range(len(urls))}

with open('relabelling.json', 'w') as outfile:
    json.dump(json_out, outfile)