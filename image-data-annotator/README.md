# Annotating using this software pipeline

## Requirements and setup

- React, version 17 and above
    - You can download this at the following link: https://react-cn.github.io/react/downloads.html

<br/>
<br/>

## Setting up the Google API

To set up your Google Spreadsheets API project, perform the following steps. Note: these steps are just taken from the following youtube video, from minute 8 to minute 20: https://www.youtube.com/watch?v=yhCJU4aqMb4

1. Create a google sheets file in your google drive. Share this file with all associated collaborators.
2. Go to console.cloud.google.com, and sign in with your gmail.
3. Go to the drop-down selection, and click "New Project"
4. Give the project a name, and create it
5. Click "Enable API services"
6. Type in 'sheets', select the "Google sheets API", and enable it
7. Go back to your project, and select "Create credentials"
8. Select the 'Google sheets API'  
9. For "What data will you be accessing", select "User data"
10. Fill out the form for "OAuth Consent Screen" as normal
11. Skip "Scopes" section
12. For "OAuth Client ID", select "Web Application"
13. For "Authorized JavaScript Origins", enter the URL of the domain you are hosting this web application on. For example, if you're hosting through GitHub/GitLab, it should be something like ______.github.io. Also add http://localhost:3000 for testing. Add the same domains for your redirect URIs.
14. Now you can generate a CLIENT_ID. Copy this, and paste it in the config.js file.
15. Click "Create Credentials" again, and this time select "API key"
16. Copy the API key, and paste it in the config.js file.
17. On the Google Cloud Platform home screen, click on "OAuth consent screen"
18. Go to "Test users"
19. Add all annotators to this list

Note: in general, you should not have these secrets on your frontend (they should be on a server). However you can specify configurations for your google API project that make it safe to do so.
<br/>
<br/>

You have now set up your web app to communicate with your google sheets!

## Running the app

After executing the steps above, type `npm i` in your terminal (this installs all dependencies). Then, you can start the app anytime by entering `npm start`.

## Deployment

You can use any service you want to deploy your image annotation app. The simplest and easiest setup is to deploy using github / gitlab pages. A guide on that can be found at the following link: https://www.youtube.com/watch?v=2hM5viLMJpA


## config.js

This is where you can configure your specific application. In the config dictionary:

    "appTitle": The title of the application shown at the top of the page,
    "adminEmails": A list of emails who are admins of the app,
    "sheetsConfig": {
        "API_KEY": The API key defined above,
        "CLIENT_ID": The client ID defined above,
        "SCOPE": The scope of the app (should be left as default value)
    },
    "users": {
        "SAMPLEUSER@gmail.com": {
                "SPREADSHEET_ID": The ID of the spreadsheet to be mutated,
                "SPREADSHEET_RANGE": The page in that spreadsheet
        },
        "default": default settings of the object described above
    },
    "annotationRules": An object of annotation rules which describe how things should be annotated.,
    "referenceList": A list of img URLs that you can use for reference,
    "imgHeight": The height of the annotated image,
    "imgWidth": The width of the annotated image,
<br/>
<br/>


## App features

1. All the admins can execute a "Flag as uncertain" action on any annotated image. All uncertain images will shown to the other annotators first.
2. Side-by-side annotation rules
3. Side-by-side reference images
4. Ability to go back and re-annotate images
5. Integration with Google Sheets for easy csv processing