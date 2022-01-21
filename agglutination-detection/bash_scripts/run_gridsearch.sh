
for i in {5..19}; do
    python experiment_runner.py --config_file ./configs/ml_test_config_$i.yml --experiment_dir ./_full_tray_imgs/20210708_13_33/experiments_$i --img_folder ./_full_tray_imgs/20210708_13_33 --ml_model_type regressor --resave true;
done;