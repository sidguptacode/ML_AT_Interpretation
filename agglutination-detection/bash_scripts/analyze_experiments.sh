img_folders_dir=$1;
config_file=$2;
model_type=ngboost

# Next, begin processing all trays (that haven't already been processed yet).
# declare -a models=("classifier" "regressor")
files=($img_folders_dir/*)
for ((i=${#files[@]}-1; i>=0; i--)); do
    img_folder="${files[$i]}";
# for img_folder in $img_folders_dir/*;
# do
    # for model_type in "${models[@]}" 
    # do
        # If we have not processed this tray before:
        if ! grep -q $img_folder ./history/processed_folders.txt; then
            # Check that this tray folder has a meta.yml file. If not, record that there's no metadata in this folder and skip it.
            if [ ! -f $img_folder/meta.yml ]; then
                echo "Could not find a meta.yml file in {$img_folder}. Skipping."
                echo $img_folder >> ./history/no_meta_yml.txt;
                continue;
            fi
            echo "Processing wells in {$img_folder}";
            # Process the tray
            python ./experiment_runner.py --config_file $config_file --experiment_dir $img_folder/experiments_$model_type --img_folder $img_folder --ml_model_type $model_type --resave false --plot_smoothed false --plot_ngboost false
            # python experiment_runner.py --config_file $config_file --experiment_dir $img_folder/experiments_gt --img_folder $img_folder --ml_model_type regressor --resave false --plot_smoothed true; 
            # python experiment_runner.py --config_file $config_file --experiment_dir ngboost/experiments --img_folder $img_folder --ml_model_type ngboost --resave false --plot_smoothed false; 
            # Record that it's been processed
            echo $img_folder >> ./history/processed_folders.txt;
        fi
        echo "Finished processing wells in {$img_folder}";
    # done;
done;