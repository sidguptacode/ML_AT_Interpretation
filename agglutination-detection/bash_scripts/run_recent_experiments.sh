
for tray in 20210802_14_03 20210730_15_50 20210730_15_45 20210730_13_39 20210729_19_35;
do
    python experiment_runner.py --config_file ./configs/ml_test_config.yml --experiment_dir ./_full_tray_imgs/$tray/experiments_ --img_folder ./_full_tray_imgs/$tray --ml_model_type regressor --resave true;
done;