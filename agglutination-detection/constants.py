MAX_PROCESSES=8
KEY_STATES = ['original', 'remove_artificial_well_boundary', 'apply_morph_close', 'apply_global_thresholding']

# The minimum area for a blob to be considered a blob.
MIN_BLOB_AREAS = {
    # "COVID": 200, # NOTE: Relative to 181x181 size
    "COVID": 20, # NOTE: Relative to 181x181 size
    "BLOOD": 625 # NOTE: Relative to 250x500 size
}

# # The minimum area for a blob to be considered a large blob
LARGE_BLOB_AREAS = {
    "COVID": 400, # NOTE: Relative to 181x181 size
    "BLOOD": 3750 # NOTE: Relative to 250x500 size
}

# The minimum area for a background to be considered "a background".
# This is only used for blood tests.
MIN_PIXEL_CONSIDERATION_AREAS = {
    "BLOOD": 1500, # NOTE: Relative to 250x500 size,
    "COVID": 10
}

IMGWS = {
    "COVID": 181,
    "BLOOD": 250
}

IMGHS = {
    "COVID": 181,
    "BLOOD": 500
}

NUM_CLASSES = 5
CONTRAST_INCR_FACTOR = 8

OWNCLOUD_PUBLIC_LINK = 'https://cloud.digiomics.com/index.php/s/vddc0Nq9CP3ereI'

ANNOTATOR_COLS = ['B', 'C', 'D', 'E']

WARMSTART_MODELS = ['RandomForestRegressor', 'MLPRegressor', 'GradientBoostingRegressor']