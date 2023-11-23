#! /bin/bash

for DATA in boxes_6dof # shapes_6dof dynamic_6dof
do
  python dataset_tools/convert_uslam_format.py --input_dir="/data/RESULT/ecd_posyaw2/desktop/uslam_event_imu/desktop_uslam_event_imu_${DATA}"
done
