import CONFIG from '../../config.js';

const getSlidesList = (files, wellName) => {
    var slidesList = [];
    for (var index = 0; index < files.length; index++) {
        var file = files[index];
        slidesList.push(
            <div>
                <div className="wellReferenceLabel">
                    <div className="wellName">
                        {wellName}
                    </div>
                    <div className="numbertext">{"" + index + "/" + files.length}</div>
                </div>
                <img src={file} style={{height: Math.floor(CONFIG['imgHeight'] / 4), width: Math.floor(CONFIG['imgWidth'] / 4)}}/>
            </div>
        );
    }
    return slidesList;
}

var referenceList = CONFIG["referenceList"];
var referenceLabels = Object.keys(referenceList);
var slideLists = [];
referenceLabels.forEach((label) => {
    var files = referenceList[label]
    slideLists.push(getSlidesList(files, label))
})

export default slideLists;
