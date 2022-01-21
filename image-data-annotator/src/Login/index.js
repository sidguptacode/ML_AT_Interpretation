
import React, { useEffect } from 'react';
import { gapi } from 'gapi-script'
import {SIDEBAR_WELCOME_MESSAGE, SIDEBAR_ERROR_MESSAGE} from '../utils/constants.js';
import {API_KEY, CLIENT_ID, SCOPE} from '../utils/constants.js'
import CONFIG from "../config";

import './styles.css';

function Login(props) {

    useEffect(() => {
        handleClientLoad();
    }, []);

    const handleClientLoad = () => {
        gapi.load('client:auth2', initClient);
    }
    
    const initClient = () => {
        gapi.client.init({
            'apiKey': API_KEY,
            'clientId': CLIENT_ID,
            'scope': SCOPE,
            'discoveryDocs': ['https://sheets.googleapis.com/$discovery/rest?version=v4'],
        }).then(() => {
            gapi.auth2.getAuthInstance().isSignedIn.listen(updateSignInStatus);
            updateSignInStatus(gapi.auth2.getAuthInstance().isSignedIn.get());
        });
    }

    const updateSignInStatus = (isSignedIn) => {
        if (isSignedIn) {
            props.setGoogleUser(gapi.auth2.getAuthInstance().currentUser.get());
        } else {
            props.setGoogleUser(null);
        }
    }

    const handleSignInClick = (event) => {
        gapi.auth2.getAuthInstance().signIn();
    }

    const handleSignOutClick = (event) => {
        gapi.auth2.getAuthInstance().signOut().then(() => {
            updateSignInStatus(gapi.auth2.getAuthInstance().isSignedIn.get());
            props.setSidebarMsg(SIDEBAR_WELCOME_MESSAGE)
        });
    }

    return (
        <div className="sidebar">
            <p>
            {CONFIG["appTitle"]}
            </p>
            <div style={{color: (props.sidebarMsg == SIDEBAR_ERROR_MESSAGE ? "red" : "black")}}>
                {props.sidebarMsg}
            </div>
            {
                props.sidebarMsg == SIDEBAR_WELCOME_MESSAGE || props.sidebarMsg == null
                ?
                <button className="userButton" onClick={handleSignInClick}>Sign in with Google</button>
                :
                <button className="userButton" onClick={handleSignOutClick}>Sign out</button>
            }
        </div>
    );
};

export default Login;
