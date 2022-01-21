
indiv_imgs_dir=$1;
labelling_dir=`pwd`;

# Creates Gitlab URLs for images
cd $indiv_imgs_dir;

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
