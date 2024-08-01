import streamlit as st
import json
from googleapiclient import discovery
from google.oauth2 import service_account

class GooglesheetUtils:
    spreadsheet_id = st.secrets['SHEET_ID']
    def __init__(self) -> None:

        service_account_json = st.secrets["spreadsheet"]['service_account']
        self.credentials = service_account.Credentials.from_service_account_info(
            service_account_json,
            scopes = ['https://www.googleapis.com/auth/spreadsheets']
        )
        self.service = discovery.build('sheets', 'v4', credentials=self.credentials)
    def append_data(self, values, range_name) -> None:
        print(self.spreadsheet_id)
        request = self.service.spreadsheets().values().append(
            spreadsheetId = self.spreadsheet_id,
            valueInputOption = 'USER_ENTERED',
            includeValuesInResponse = True,
            range = range_name,
            body = {
                'majorDimension': 'ROWS',
                'values': values
            }
        )
        response = request.execute()
        print(response)
