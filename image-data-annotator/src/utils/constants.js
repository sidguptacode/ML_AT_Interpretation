import CONFIG from "../config";

export const SIDEBAR_WELCOME_MESSAGE = "Welcome! Please sign in to begin annotating.";
export const SIDEBAR_ERROR_MESSAGE = "Your gmail account does not have access to use this webapp. Please contact " + CONFIG["adminEmails"][0] + " for access.";
export const SIDEBAR_LOGGEDIN_MESSAGE = (googleUser) => {
    if (googleUser == null) {
        return "";
    }
    return "Signed in as: " + googleUser.getBasicProfile().getEmail();
}

export const API_KEY = CONFIG['sheetsConfig']['API_KEY']
export const CLIENT_ID = CONFIG['sheetsConfig']['CLIENT_ID']
export const SCOPE = CONFIG['sheetsConfig']['SCOPE']


export const getSpreadsheetConfig = (googleUser) => {
    var userEmail = googleUser.getBasicProfile().getEmail();
    if (!(userEmail in CONFIG['users'])) {
        userEmail = "default";
    }
    const SPREADSHEET_ID = CONFIG['users'][userEmail]['SPREADSHEET_ID'];
    const SPREADSHEET_RANGE = CONFIG['users'][userEmail]['SPREADSHEET_RANGE'];
    const URL_SUBSTRINGS_TO_ANNOTATE = CONFIG['users'][userEmail]['URL_SUBSTRINGS_TO_ANNOTATE']; // TODO: Fix this!
    const FRAME_TYPE_TO_ANNOTATE = CONFIG['users'][userEmail]['FRAME_TYPE_TO_ANNOTATE']; // TODO: Remove this!
    var spreadsheet_config = [SPREADSHEET_ID, SPREADSHEET_RANGE, URL_SUBSTRINGS_TO_ANNOTATE, FRAME_TYPE_TO_ANNOTATE];
    return spreadsheet_config;
}
