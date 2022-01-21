
last_dir=$(basename `pwd`)
if [ "$last_dir" != "labelling" ]; then 
    echo "This script must be run in the labelling directory."
    return; 
fi

indiv_well_imgs_dir=$1;
labelling_dir=`pwd`;

# TODO: Our individual well images are stored in a Gitlab repo (https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images).
# In the future we'll migrate to ownCloud, but for now, Gitlab gives fast GET requests in the webapp.
cd $indiv_well_imgs_dir;

# Get all indiv_well_imgs we're about to upload
git ls-files --others --exclude-standard > $labelling_dir/files_uploaded.txt
git lfs install;
git lfs track ../indiv_well_imgs;
git add ../indiv_well_imgs;
git commit -m "Update";
git push;

# Now update the spreadsheet with the gitlab URLs
cd $labelling_dir;
python update_spreadsheet_with_urls.py --files_uploaded ./files_uploaded.txt --prefix gitlab;
