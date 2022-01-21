from __future__ import print_function
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import argparse
import numpy as np
from spreadsheet_constants import SCOPES, SPREADSHEET_ID, RANGE_NAME
import sys
sys.path.insert(0,'..')
import time
from tqdm import tqdm

def get_spreadsheet_service(token_path=f".{os.sep}token.json", creds_path=f".{os.sep}credentials.json"):
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    service = build('sheets', 'v4', credentials=creds)
    return service

def append_urls_to_spreadsheet(img_url_list, token_path=f".{os.sep}token.json", creds_path=f".{os.sep}credentials.json"):
    """
        Appends a list of URLs to our google spreadsheet, if they're not already in there.
    """
    service = get_spreadsheet_service(token_path, creds_path)

    ## Call the sheets API to upload the URL list.
    # How the input data should be interpreted.
    value_input_option = 'USER_ENTERED'
    # How the input data should be inserted.
    insert_data_option = 'INSERT_ROWS'
    # Get the current values in the spreadsheet to check for duplicates.
    existing_urls = read_spreadsheet(service)[0]
    values_to_append = []
    for url in img_url_list:
        if url not in existing_urls:
            values_to_append.append([url])
    value_range_body = {
        "majorDimension": "ROWS",
        "values": values_to_append
    }
    request = service.spreadsheets().values().append(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME, valueInputOption=value_input_option, insertDataOption=insert_data_option, body=value_range_body)
    response = request.execute()


def read_spreadsheet(service):
    """
        Reads all the values already in the spreadsheet.
    """
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID,
                                range=RANGE_NAME).execute()
    values = result.get('values', [])
    # Reformat the read values, so that the zero'th index represents all values in the first column of the spreadsheet.
    values = np.array(values).T.tolist()

    if not values:
        raise Exception('Could not read from the spreadsheet.')
    else:
        return values
    