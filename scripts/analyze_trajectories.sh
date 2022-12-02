#! /bin/bash

python analyze_trajectories.py \
  ../analyze_trajectories_config/uslam_event_imu.yaml \
  --output_dir="/data/RESULT/ecd_posyaw" \
  --results_dir="/data/RESULT/ecd_posyaw" \
  --platform desktop \
  --odometry_error_per_dataset \
  --plot_trajectories \
  --rmse_table --rmse_boxplot --mul_trials=8 --png
