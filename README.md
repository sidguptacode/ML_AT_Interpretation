# ML Analysis with agglutination-detection

## Requirements and setup

- Python 3.8

We'll describe our setup using Python virtual environments:

1. Create and activate your Python virtual environment.

      a. For the commands to do this, see: https://docs.python.org/3/library/venv.html

2. Install all package dependencies using `pip install -r requirements.txt`.

3. If you plan on committing code, please update the .gitignore so that it includes your virtual environment folder.

<br/>
<br/>

## Downloading folders from ownCloud

You can do this through the ownCloud interface. But we also have written a script to do so.

To download the last 10 trays from ownCloud, simply run:
`python download_owncloud_tray_imgs.py --num_trays 10`

If you want to download just one tray, run:
`python download_owncloud_tray_imgs.py --folder1 20210603_11_58`

<br/>
<br/>

## Running on a single image folder

Let's say our image folder is at: `./_full_tray_imgs/20210603_11_58`

We can run the following command:

`python experiment_runner.py --config_file ./configs/ml_test_config.yml --experiment_dir ./_full_tray_imgs/20210603_11_58/experiments --img_folder ./_full_tray_imgs/20210603_11_58 --ml_model_type ngboost`

Don't worry if you see warning messages -- those are normal.

<br/>
<br/>

## Running a batch of image folders

Let's say all of our image folders are located within : `./_full_tray_imgs`.

To run our analysis on every image folder in a directory, we can use the following command:

`bash analyze_experiments.sh ./_full_tray_imgs ./configs/ml_test_config.yml`

This bash script searches in a local `./history/processed_folders.txt`, so it won't process folders that have already been analyzed. If you want to re-analyze a folder, simply remove it's name from the `./history/processed_folders.txt` file.

<br/>
<br/>


## `ml_test_config.yml`

If you want to plot the tray at different processing steps (e.g, after adaptive thresholding), you can modify the `steps` attribute in the `plot_well_matrix` dictionary. See the in-line comments there for more details.

<br/>
<br/>



## Clearing storage space

For each processed image folder, we save it's Python data structure in `./_saved_well_matrices`. If you are done processing an image folder, and don't plan on processing it again, you can delete it's `.h5` file inside this folder.


<br/>
<br/>
<br/>
<br/>

# Other code folders

NOTE: If you are only interested in running our algorithm, you can disregard the below. However, if you want to understand how the pipeline works, from updating the website and pulling labels, to training ML models, see the below
<br/>
<br/>
## `experiment_runner.py` code structure
The project starts with `experiment_runner.py`. This file reads provided `ml_test_config.yml` config, and generates experiment results in an `experiments` folder. The code is modularized across the following folders:

- `feature_extraction`: contains the code that does all image processing, and creates feature vectors for machine learning
- `plotting_and_visuals`: code for plotting metrics of the tray over time (rsd, blob area, agglutination score), as well as visualizing the wells
- `training_ml`: code which trains the best-performing ML models as a .pkl in the main directory (explained more below). 
- `testing_ml`: contains helper functions called when using the ML model for evaluation. 
- `well_matrix_creation`: all datastructure code regarding how we store individual wells and well metadata
- `configs`: contains the `ml_test_config.yml`
- `history`: contains .txt files which are read by `analyze_experiments.sh` to avoid duplicate work

<br/>
<br/>

## Updating the website

To update the website, we use `download_owncloud_tray_imgs.py` and `download_process_upload.sh`. We do the following:

1. Pick the trays you want to download from ownCloud (e.g, `20210603_11_58`), and download them using `download_owncloud_tray_imgs.py`

2. Clone the `agglutination-data-storage` repo from Gitlab (https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images)

2. Run `download_process_upload.sh`, which partition the downloaded wells into images, saves each image as a .png in the `agglutination-data-storage` repo, pushes the new images, and updates the spreadsheet

<!-- NOTE: For now, we have a seperate repo to store the images.
        This is the only step that changes to have a perfectly replicable system.
        If I can get ownCloud to work fast, then this step effectively changes to:
        "upload local indiv_well_imgs folder to ownCloud, get URLs, update spreadsheet".

        Since it's just me for now, I can use gitlab. But when I do the handoff, I'll switch to ownCloud
        & make it fast.

        Although, if no labelling needs to be done, then this block of steps can be entirely ignored.
-->

4. The following is how we run the command: `bash download_process_upload.sh ../_full_tray_imgs ../../agglutination-data-storage/indiv_well_imgs`

<br/>
<br/>

## ML Training and testing

We'll mostly focus on the code inside the `training_ml` folder. 

`pull_labels.py` will pull all labels from the spreadsheet, and analyze those labels. It gets confusion matrices from all annotators, and automatically marks points of disagreement in a .txt with their URLs

`create_dataset.py` creates a dataset from the labels, doing the following:

1. Downloads the non-present full_tray_images that correspond to the labelled images.

2. Process the labelled trays into WellMatrix objects. Assign labels to the interior Well objects.

3. Save the labelled WellMatrix objects as .h5, with some metadata that denotes they're labelled.

4. Compute feature vectors, and match those with the labels. Save the raw data as .npz.

`train.py` is used to train various ML models, and it saves the best performing one as a .pkl to be used by `experiment_runner.py`.

<br/>
<br/>


# Dataset annotation with image-data-annotator

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