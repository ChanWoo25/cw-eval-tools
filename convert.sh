#! /bin/bash

# for DATA in church_soft church_stand dark_easy dark_hard light_easy2 ; do
#   sudo chmod 777 -R /data/results/DSR_Vins/${DATA}/
#   python3 scripts/dataset_tools/convert_uslam_format.py --single-file /data/results/DSR_Vins/${DATA}/vio.csv
# done

#cvlab_soft church_soft church_stand dark_easy dark_hard light_easy2
for DATA in light_hard ; do
  sudo chmod 777 -R /data/results/DSR_Vins/${DATA}/
  python3 scripts/dataset_tools/convert_uslam_format.py --single-file /data/results/DSR_Vins/${DATA}/vio.csv
  python3 /home/leecw/Reps/namu_trajectory_evaluation/utils/analyizeEndToEnd.py \
    --input_dir /data/results/DSR_Vins/${DATA}
done
