import { gapi } from 'gapi-script'
import {partitionIndices, mapAnnotatorIndexToCol, getSelectURLIndices} from './helpers.js';
import {SIDEBAR_ERROR_MESSAGE, SIDEBAR_LOGGEDIN_MESSAGE} from './constants.js';
import {getSpreadsheetConfig} from './constants.js'


const filterUrl = (url, frameTypeToAnnotate, currTraysToAnnotate) => {
    // Get the frame number from the URL
    var imgURL_ = url.split('_');
    imgURL_ = imgURL_[imgURL_.length - 1].split('.')[0];
    var frameNum = parseInt(imgURL_);
    if (frameTypeToAnnotate == 'mod_5' && frameNum % 5 != 0) {
        return false;
    }
    // Check if the URL belongs to one of the CURR_TRAYS_TO_ANNOTATE
    if (currTraysToAnnotate.length == 0) { // annotating all trays
        return true;
    }
    for (var k = 0; k < currTraysToAnnotate.length; k++) {
      if (url.includes(currTraysToAnnotate[k])) {
          return true;
      }
    }
    return false;
}


export const readSpreadsheet = (spreadsheetId, spreadsheetRange) => {
    var params = {
        // The ID of the spreadsheet to retrieve data from.
        spreadsheetId: spreadsheetId,
        // The A1 notation of the values to retrieve.
        range: spreadsheetRange,
        majorDimension: 'COLUMNS'
    };
    var request = gapi.client.sheets.spreadsheets.values.get(params);
    return request;
}


export const initWithSpreadsheetData = (googleUser, setSidebarMsg, setImgURLList, setAnnotatedIndexList, setUnannotatedIndexList, setUncertainIndexList, setAnnotatorIndex) => {
    if (googleUser == null) {
        console.log("googleUser is null, not reading the spreadsheet.");
        return;
    }
    console.log("Reading the spreadsheet.");
    var spreadsheet_config = getSpreadsheetConfig(googleUser);
    var spreadsheetId = spreadsheet_config[0];
    var spreadsheetRange = spreadsheet_config[1];
    var currTraysToAnnotate = spreadsheet_config[2];
    var frameTypeToAnnotate = spreadsheet_config[3];

    var request = readSpreadsheet(spreadsheetId, spreadsheetRange);
    request.then((response) => {
        setSidebarMsg(SIDEBAR_LOGGEDIN_MESSAGE(googleUser));
        // NOTE: allURLList contains the header "Well Img URL".
        // But it's okay, because annotationIndexList will include index 0 always.
        var allURLList = response.result.values[0];
        // To make things faster, we will mutate unannotatedIndices such that
        // it only contains indices of the trays and frames we want.
        // var selectURLList = allURLList;
        var selectURLList = allURLList.filter(url => filterUrl(url, frameTypeToAnnotate, currTraysToAnnotate));
        console.log("Length of filtered urls: " + selectURLList.length)
        // TODO: Revert when we're annotating trays!
        // var selectURLList = allURLList;
        setImgURLList(allURLList);

        var annotationLists = response.result.values.slice(1);
        var localAnnotatorIndex = -1;
        annotationLists.forEach((annotationList, i) => {
            var annotatorID = annotationList[0];
            if (annotatorID == googleUser.getBasicProfile().getEmail()) {
                localAnnotatorIndex = i + 1;
            }
        });

        // Case where this is a new annotator. Here, we write a column to the end of the spreadsheet.
        if (localAnnotatorIndex == -1) {
            localAnnotatorIndex = writeAnnotatorToSpreadsheet(googleUser, response.result.values[0], selectURLList, annotationLists.length, setAnnotatedIndexList, setUnannotatedIndexList, spreadsheetId, spreadsheetRange);
        } else {
            // unannotatedIndices have values of either -1 or -2. Any other value means it's an annotated index.
            var annotatorCol = response.result.values[localAnnotatorIndex]
            var indexPartition = partitionIndices(annotatorCol, "-1", "-2");
            setAnnotatedIndexList(indexPartition[0]);
            console.log("indexPartition[0].length" + indexPartition[0].length);
            console.log("indexPartition[1].length" + indexPartition[1].length);
            // Refactor unannotatedIndices and uncertainIndices so it keeps only the indices from the selectedURLList.
            var unannotatedIndices = getSelectURLIndices(selectURLList, allURLList, indexPartition[1]); // Still in the allURLList index domain
            console.log("unannotatedIndices.length" + unannotatedIndices.length);
            // From the unannotatedIndices, seperate the ones with value -2 (which means they're labelled as uncertain by Sid, 
            //  and should be annotated before the -1 valued indices)
            var unannotatedVals = unannotatedIndices.map(i => annotatorCol[i]); // TODO: This may be wrong, if we do it after line 74
            var uncertainIndexPartition = partitionIndices(unannotatedVals, "-1", "-1");
            var uncertainIndices = uncertainIndexPartition[0].map(i => unannotatedIndices[i]);
            var unannotatedIndices = uncertainIndexPartition[1].map(i => unannotatedIndices[i]);
            setUncertainIndexList(uncertainIndices);
            setUnannotatedIndexList(unannotatedIndices);
        }
        setAnnotatorIndex(localAnnotatorIndex);
    }, (reason) => {
        console.error('error: ' + reason.result.error.message);
        setSidebarMsg(SIDEBAR_ERROR_MESSAGE);
    });
}


export const writeAnnotatorToSpreadsheet = (googleUser, imgURLList, selectURLList, numAnnotators, setAnnotatedIndexList, setUnannotatedIndexList, spreadsheetId, spreadsheetRange) => {

    var valuesToAppend = [[googleUser.getBasicProfile().getEmail()]];
    for (var i = 1; i < imgURLList.length; i++) {
        valuesToAppend.push(["-1"]);
    }

    var newAnnotatorIndex = numAnnotators + 1;

    if (newAnnotatorIndex > 20) {
        alert("Too many annotators; please notify sid.gupta@mail.utoronto.ca.")
        return;
    }

    var annotatorCol = mapAnnotatorIndexToCol(newAnnotatorIndex);

    var params = {
        // The ID of the spreadsheet to update.
        spreadsheetId: spreadsheetId,  // TODO: Update placeholder value.

        // The A1 notation of a range to search for a logical table of data.
        // Values will be appended after the last row of the table.
        // range: SPREADSHEET_RANGE + "!" + annotatorCol + "1" + ":" + annotatorCol + imgURLList.length,  // TODO: Update placeholder value.
        range: spreadsheetRange + "!" + annotatorCol + "1",  // TODO: Update placeholder value.

        // How the input data should be interpreted.
        valueInputOption: 'USER_ENTERED',
        // How the input data should be inserted.
        // insertDataOption: 'INSERT_ROWS',  // TODO: Update placeholder value.
    };

    var valueRangeBody = {
        "majorDimension": "ROWS",
        "values": valuesToAppend
    };

    var request = gapi.client.sheets.spreadsheets.values.update(params, valueRangeBody);
    request.then(function(response) {
        // unannotatedIndices is all indices except 0.
        var unannotatedIndices = Array.from({length: valuesToAppend.length}, (x, i) => i);
        unannotatedIndices.shift();
        unannotatedIndices = getSelectURLIndices(selectURLList, imgURLList, unannotatedIndices);
        setAnnotatedIndexList([0]);
        setUnannotatedIndexList(unannotatedIndices);
    }, function(reason) {
        console.error('error: ' + reason.result.error.message);
        return -1;
    });
    return newAnnotatorIndex;
}
