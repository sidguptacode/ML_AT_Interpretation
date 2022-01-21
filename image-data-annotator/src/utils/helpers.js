
export function getAllIndices(arr, val) {
    var indices = [], i = -1;
    while ((i = arr.indexOf(val, i+1)) != -1){
        indices.push(i);
    }
    return indices;
}


export function partitionIndices(arr, val1, val2) {
    var annotatedIndices = [];
    var unannotatedIndices = [];
    for (var i = 0; i < arr.length; i++) {
        if (arr[i] === val1 || arr[i] === val2) {
            unannotatedIndices.push(i);
        } else {
            annotatedIndices.push(i);
        }
    }
    var output = [annotatedIndices, unannotatedIndices];
    return output;
}

export function mapAnnotatorIndexToCol(annotatorIndex) {
    return String.fromCharCode("A".charCodeAt(0) + annotatorIndex);
}

export function getSelectURLIndices(selectURLList, allURLList, allUnannotatedIndices) {
    var unannotatedIndices = [];
    for (var i = 0; i < selectURLList.length; i++) {
        var urlIndex = allURLList.indexOf(selectURLList[i]);
        if (allUnannotatedIndices.includes(urlIndex)) {
            unannotatedIndices.push(urlIndex)
        }
    }
    return unannotatedIndices;
}