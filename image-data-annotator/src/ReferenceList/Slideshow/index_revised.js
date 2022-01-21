import React, { useState, useEffect } from 'react';
import './styles.css';
import slideLists from './slide_lists.js';

// TODO: This is a fix that allows N reference label images (currently we only support 6).
// Currenlty this file is unused in production.

function Slideshow() {
    const [currSlideInds, setCurrSlideInds] = useState([]);
    useEffect(() => {
        var initCurrSlideInds = [];
        for (var i = 0; i < slideLists.length; i++) {
            initCurrSlideInds.push(0);
        }
        setCurrSlideInds(initCurrSlideInds);
    }, [])

    const getCurrSlides = (slideLists, currSlideInds, setCurrSlideInds) => {
        var currSlides = [];
        for (var i = 0; i < currSlideInds.length; i++) {
            var currSlideDeck = slideLists[i];
            var currSlideInd = currSlideInds[i];
            var currSlide = [];
            currSlide.push(currSlideDeck[currSlideInd]);
            currSlide.push(
                <button className="prev" onClick={() => {
                    var newCurrSlideInds = currSlideInds.slice(0);
                    newCurrSlideInds[i] = currSlideInd - 1;
                    setCurrSlideInds(newCurrSlideInds)
                }}>❮</button>
            );
            currSlide.push(
                <button className="next" onClick={() => {
                    var newCurrSlideInds = currSlideInds.slice(0);
                    newCurrSlideInds[i] = currSlideInd - 1;
                    setCurrSlideInds(newCurrSlideInds)
                }}>❯</button>
            );
            currSlides.push(currSlide);
        }
        
        return currSlides;
    }

    var currSlides = getCurrSlides(slideLists, currSlideInds, setCurrSlideInds);
    var halfNumSlides = currSlides.length / 2;
    if (halfNumSlides % 1 != 0) {
        currSlides.append([])
    }
    halfNumSlides = Math.floor(halfNumSlides);
    var slideComponents = []
    for (var i = 0; i < halfNumSlides; i++) {
        slideComponents.push(
            <React.Fragment>
                <div className="firstRow">
                    <div className="container">
                        <div className="mySlides">
                            {currSlides[2 * i]}
                        </div>
                    </div>
                    <div className="container">
                    <div className="mySlides">
                            {currSlides[2 * i + 1]}
                        </div>
                    </div>
                </div>
                <br/>
            </React.Fragment>
        )
    }

    return (
        <div className='slideshows'>
            {
                slideComponents
            }
{/*             
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
            </div> */}
        </div>
    );
};

export default Slideshow;

