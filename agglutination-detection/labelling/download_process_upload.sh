
last_dir=$(basename `pwd`)
if [ "$last_dir" != "labelling" ]; then 
    echo "This script must be run in the labelling directory."
    return; 
fi

full_tray_imgs=$1;
# PRECONDITION: indiv_well_imgs_dir points to a folder in a Gitlab repo 
#   (https://gitlab.com/wheeler-microfluidics/agglutination-image-db-indiv-well-images/-/tree/master/indiv_well_imgs)
indiv_well_imgs_dir=$2;
segmented_trays="./_segmented_trays.txt"

# First, download two new tray_images from owncloud, that aren't already stored locally
# TODO: The download part needs to be done seperately, as long as A1_top is not in the ymls by default.
# echo "Downloading images from opencloud"
# python ../download_owncloud_tray_imgs.py --save_dirpath $full_tray_imgs --num_trays 2;

# Next, begin processing all trays (that haven't already been processed yet) into individual well images.
for tray_imgs_dirpath in $full_tray_imgs/*;
do
    # If we have not processed this tray before:
    if ! grep -q $tray_imgs_dirpath $segmented_trays; then
        # Check that this tray folder has a meta.yml file. If not, record that there's no metadata in this folder and skip it.
        if [ ! -f $tray_imgs_dirpath/meta.yml ]; then
            echo "Could not find a meta.yml file in {$tray_imgs_dirpath}"
        fi
        echo "Processing wells in {$tray_imgs_dirpath}";
        # Process the tray
        python ./process_trays_into_indiv_wells.py --tray_imgs_dirpath=$tray_imgs_dirpath  --indiv_wells_dir $indiv_well_imgs_dir;
        # Record that it's been processed
        echo $tray_imgs_dirpath >> $segmented_trays;
    fi
    echo "Finished processing wells in {$tray_imgs_dirpath}";
done;

bash upload_indiv_well_imgs.sh $indiv_well_imgs_dir; 