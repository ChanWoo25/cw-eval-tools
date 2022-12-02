#! /bin/bash

for DATA in boxes_6dof shapes_6dof poster_6dof dynamic_6dof
do
  python bag_to_pose.py /data/EVENT/rosbag/${DATA}.bag  /optitrack/davis --output=${DATA}_stamped_groundtruth.txt
done
