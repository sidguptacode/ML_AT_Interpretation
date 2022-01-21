import React, { useState } from 'react';
import './styles.css';
import slideLists from './slide_lists.js';

function Slideshow() {
    const [zeroInd, setZeroInd] = useState(0);
    const [oneInd, setOneInd] = useState(0);
    const [twoInd, setTwoInd] = useState(0);
    const [threeInd, setThreeInd] = useState(0);
    const [fourInd, setFourInd] = useState(0);
    const [naInd, setNaInd] = useState(0);

    var zeroSlidesList = slideLists[0];
    var oneSlidesList = slideLists[1];
    var twoSlidesList = slideLists[2];
    var threeSlidesList = slideLists[3];
    var fourSlidesList = slideLists[4];
    var naSlidesList = slideLists[5];


    const getCurrSlide = (slideList, currInd, setInd) => {
        var currSlide = [];
        currSlide.push(slideList[currInd]);
        currSlide.push(
            <button className="prev" onClick={() => setInd(currInd - 1)}>❮</button>
        );
        currSlide.push(
            <button className="next" onClick={() => setInd(currInd + 1)}>❯</button>
        );
        return currSlide;
    }
    var zeroSlide = getCurrSlide(zeroSlidesList, zeroInd, setZeroInd); 
    var oneSlide = getCurrSlide(oneSlidesList, oneInd, setOneInd); 
    var twoSlide = getCurrSlide(twoSlidesList, twoInd, setTwoInd); 
    var threeSlide = getCurrSlide(threeSlidesList, threeInd, setThreeInd); 
    var fourSlide = getCurrSlide(fourSlidesList, fourInd, setFourInd); 
    var naSlide = getCurrSlide(naSlidesList, naInd, setNaInd); 
    
    return (
        <div className='slideshows'>
            <div className="firstRow">
                <div className="container">
                    <div className="mySlides">
                        {zeroSlide}
                    </div>
                </div>
                <div className="container">
                <div className="mySlides">
                        {oneSlide}
                    </div>
                </div>
            </div>
            <br/>
            <div className="secondRow">
                <div className="container">
                <div className="mySlides">
                    {twoSlide}
                    </div>
                </div>
                <div className="container">
                <div className="mySlides">
                    {threeSlide}
                    </div>
                </div>
            </div>
            <br/>
            <div className="thirdRow">
                <div className="container">
                <div className="mySlides">
                    {fourSlide}
                    </div>
                </div>
                <div className="container">
                <div className="mySlides">
                    {naSlide}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Slideshow;

