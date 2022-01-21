
import React from 'react';
import Slideshow from './Slideshow';

function ReferenceList() {
    return (
        <>
            <p>
                Reference Well Scores
            </p>
            <div style={{color: "firebrick", paddingBottom: 20}}>
                {"Note: buttons have some delay."}
            </div>
            <Slideshow/>
        </>
    );
};

export default ReferenceList;
