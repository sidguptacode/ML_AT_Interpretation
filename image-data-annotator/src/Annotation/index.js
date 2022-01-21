
import React, { useEffect, useState } from 'react';
import { gapi } from 'gapi-script'
import {getSpreadsheetConfig} from '../utils/constants.js'
import {mapAnnotatorIndexToCol} from '../utils/helpers.js';
import { readSpreadsheet } from '../utils/spreadsheet_functions.js';
import CONFIG from '../config.js';
import './styles.css';

function Annotation(props) {
    
    const [certain, setCertain] = useState(true);
    const [inputText, setInputText] = useState("");

    useEffect(() => {
      setCertain(true);
    }, [props.currWellImg]);


    const updateIndexList = (currWellIndex, indexList, setIndexList) => {
      var indexListNew = [...indexList];
      const indexToRm = indexListNew.indexOf(currWellIndex);
      indexListNew.splice(indexToRm, 1);
      setIndexList(indexListNew);
    }

    const updateCellRequest = (annotatorCol, currWellIndex, label, spreadsheetId, spreadsheetRange) => {
      var params = {
        // The ID of the spreadsheet to update.
        spreadsheetId: spreadsheetId,  // TODO: Update placeholder value.
        // The A1 notation of the values to update.
        range: spreadsheetRange + "!" + annotatorCol + "" + (currWellIndex + 1), 
        valueInputOption: 'USER_ENTERED',
      };
      var valueRangeBody = {
        "majorDimension": "ROWS",
        "values": [[label]]
      };
      var request = gapi.client.sheets.spreadsheets.values.update(params, valueRangeBody);
      return request;
    }

    const annotate = (label) => {
        var annotatorCol = mapAnnotatorIndexToCol(props.annotatorIndex);
        var spreadsheet_config = getSpreadsheetConfig(props.googleUser);
        var spreadsheetId = spreadsheet_config[0];
        var spreadsheetRange = spreadsheet_config[1];
        if (certain == false) {
          // If this label is uncertain, notify the others to check it.
          var spreadsheetDataReq = readSpreadsheet(spreadsheetId, spreadsheetRange);
          spreadsheetDataReq.then((response) => {
            var spreadsheetCols =  response.result.values;
            for (var colInd = 1; colInd < spreadsheetCols.length; colInd++) {
              // If we are on another annotator's column, and this other annotator has not annotated this well
              console.log(spreadsheetCols[colInd][props.currWellIndex]);
              if (colInd != props.annotatorIndex && spreadsheetCols[colInd][props.currWellIndex] == -1) {
                var otherAnnotatorCol = mapAnnotatorIndexToCol(colInd);
                var updateOthersReq = updateCellRequest(otherAnnotatorCol, props.currWellIndex, "-2", spreadsheetId, spreadsheetRange);
                updateOthersReq.then(function(response) {
                  console.log("Gave other annotator a label of -2, as desired.");
                });
              }
            }
          });
        }
        var request = updateCellRequest(annotatorCol, props.currWellIndex, label, spreadsheetId, spreadsheetRange);
        request.then(function(response) {
          var annotatedIndexListNew = [...props.annotatedIndexList];
          if (!annotatedIndexListNew.includes(props.currWellIndex)) {
            annotatedIndexListNew.push(props.currWellIndex)
          }
          props.setAnnotatedIndexList(annotatedIndexListNew);

          if (props.unannotatedIndexList.includes(props.currWellIndex)) {
            updateIndexList(props.currWellIndex, props.unannotatedIndexList, props.setUnannotatedIndexList);
          } else {
            updateIndexList(props.currWellIndex, props.uncertainIndexList, props.setUncertainIndexList);
          }

        }, function(reason) {
          console.error('error: ' + reason.result.error.message);
        });
        setInputText("");
    }

    // Adjust image sizes for different datasets
    var imgH = CONFIG['imgHeight'];
    var imgW = CONFIG['imgWidth'];
    var currUserEmail = props.googleUser.getBasicProfile().getEmail();

    return (
        <>
            {
                props.currWellImg != ""
                ?
                <div>
                    <div>
                      {/* {props.currWellImg.substring(props.currWellImg.lastIndexOf('/') + 1).slice(0, -7)} */}
                      {props.currWellImg}
                    </div>
                    <img style={{height: imgH, width: imgW}} src={props.currWellImg}></img>
                    <br/>
                    <br/>
                    <div className="inputField">
                        <div/>
                        <input value={inputText} onChange={e => setInputText(e.target.value)} />
                        <div/>
                    </div>
                    <div>
                        <br/>
                        <button className="submitButton" onClick={() => annotate(inputText)} style={{backgroundColor: '#496BD9'}}>{"Submit"}</button>
                    </div>
                    { 
                         CONFIG['adminEmails'].includes(currUserEmail)
                          ?
                          <div>
                            <br/>
                            <br/>
                          <button className="uncertainButton" onClick={() => setCertain(!certain)} style={{backgroundColor: certain ? 'indianred' : 'forestgreen'}}>{certain ? "Flag as Uncertain" : "Flag as Certain"}</button>
                          </div>
                          :
                          <div/>
                    }
                    <br/>
                </div>
                :
                <div/>
            }
        </>
    );
};

export default Annotation;
