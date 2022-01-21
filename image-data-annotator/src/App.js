import React, { useEffect, useState } from 'react';
import Login from "./Login"
import Annotation from "./Annotation"
import ReferenceList from "./ReferenceList"
import History from "./History"
import { initWithSpreadsheetData } from './utils/spreadsheet_functions.js';
import {SIDEBAR_WELCOME_MESSAGE, SIDEBAR_ERROR_MESSAGE, SIDEBAR_LOGGEDIN_MESSAGE} from './utils/constants.js';
import "./App.css";


function App() {
    const [googleUser, setGoogleUser] = useState(null);
    const [annotatorIndex, setAnnotatorIndex] = useState(-1);
    const [sidebarMsg, setSidebarMsg] = useState(SIDEBAR_WELCOME_MESSAGE);
    const [imgURLList, setImgURLList] = useState([]);
    const [unannotatedIndexList, setUnannotatedIndexList] = useState([]);
    const [annotatedIndexList, setAnnotatedIndexList] = useState([]);
    const [uncertainIndexList, setUncertainIndexList] = useState([]);
    const [currWellImg, setCurrWellImg] = useState("");
    const [currWellIndex, setCurrWellIndex] = useState(-1);
    const [amountAnnotated, setAmountAnnotated] = useState(0);


    useEffect(() => {
      initWithSpreadsheetData(googleUser, setSidebarMsg, setImgURLList, setAnnotatedIndexList, setUnannotatedIndexList, setUncertainIndexList, setAnnotatorIndex);
    }, [googleUser]);

    useEffect(() => {
      setCurrWellImg(imgURLList[currWellIndex]);
    }, [currWellIndex]);

    useEffect(() => {
      setAmountAnnotated(amountAnnotated + 1);
      if (uncertainIndexList.length != 0) {
        var randIndex = uncertainIndexList[0];
      } else {
        var randIndex = unannotatedIndexList[0];
      }
      console.log("Length of unannotatedIndexList: " + unannotatedIndexList.length)
      setCurrWellIndex(randIndex);
    }, [unannotatedIndexList, uncertainIndexList]);

    return (
      <div className="app">
          <div className="sidebar">
            <Login setGoogleUser={setGoogleUser} setSidebarMsg={setSidebarMsg} sidebarMsg={sidebarMsg}/>
            {
              sidebarMsg == SIDEBAR_LOGGEDIN_MESSAGE(googleUser) && annotatorIndex != -1 && annotatorIndex != null && currWellIndex != -1 && currWellIndex != null && currWellImg != "" && currWellImg != null
              ?
              <History imgURLList={imgURLList} annotatedIndexList={annotatedIndexList} setCurrWellIndex={setCurrWellIndex}/>
              :
              <div/>
            }
          </div>
          <div className="annotation">
              {
              sidebarMsg == SIDEBAR_LOGGEDIN_MESSAGE(googleUser) && annotatorIndex != -1 && annotatorIndex != null && currWellIndex != -1 && currWellIndex != null && currWellImg != "" && currWellImg != null
              ?
                <Annotation currWellImg={currWellImg} annotatedIndexList={annotatedIndexList} setAnnotatedIndexList={setAnnotatedIndexList} 
                annotatorIndex={annotatorIndex} currWellIndex={currWellIndex} unannotatedIndexList={unannotatedIndexList} setUnannotatedIndexList={setUnannotatedIndexList} 
                uncertainIndexList={uncertainIndexList} setUncertainIndexList={setUncertainIndexList} googleUser={googleUser}/>
              :
                <div/>
              }
          </div>
          <div className="reference">
              {
              sidebarMsg == SIDEBAR_LOGGEDIN_MESSAGE(googleUser)
              ? 
                <ReferenceList />
              : 
                <div/>
              } 
          </div>
      </div>
    );
}

export default App;
