
import React  from 'react';
import './styles.css';
import CONFIG from '../config';

function History(props) {
    var annotatedImgURLs = [];
    for (var i = 1; i < props.annotatedIndexList.length; i++) {
        var imageIndex = props.annotatedIndexList[i];
        var imageURL = props.imgURLList[imageIndex];
        var imageName = i + ". " + imageURL.substring(imageURL.lastIndexOf('/') + 1);
        annotatedImgURLs.push(
            <HistoryItem imageIndex={imageIndex} imageName={imageName} setCurrWellIndex={props.setCurrWellIndex}/>
        )
    }

    annotatedImgURLs.reverse();

    var annotationRules = CONFIG["annotationRules"];

    return (
        <div>
            <br/>
            {"History - " + (annotatedImgURLs.length) + " images annotated."}
            <br/>
            <br/>
            <br/>
            <div className='history'>
                {annotatedImgURLs}
            </div>
            <br/>
            <AnnotationRules rulesDict={annotationRules}/>
        </div>
    );
};

function HistoryItem(props) {

    const updateWellIndex = (index) => {
        console.log("updated to " + index);
        props.setCurrWellIndex(index)
    }

    var imgName = props.imageName.slice(0, -6);

    return (
        <div className="historyItem">
            <div className="imageName">
                {imgName}
            </div>
            <div/>
            <button className="redoButton" onClick={() => {updateWellIndex(props.imageIndex)}}>
                {"Redo"}
            </button>
        </div>
    )
}


function AnnotationRules(props) {
    var rulesList = [];
    for (const label in props.rulesDict) {
        var rules = props.rulesDict[label]; 
        rulesList.push(
            <>
            <div className="labelHeader">
                {label}
            </div>
            <div className="ruleList">
                <RuleList rules={rules} />
            </div>
            </>
        );
    }

    return (
        <div >
            <div className="rulesHeader">
                {"Annotation rules:"}
            </div>
            {rulesList}
        </div>
    )
}

function RuleList(props) {
    var ruleItems = [];
    for (const i in props.rules) {
        ruleItems.push(
            <div className="ruleItem">
                {props.rules[i]}
            </div>
        );
    }

    return (
        ruleItems
    )
}



export default History;
