// Copyright (c) 2023 Chanwoo Lee, All Rights Reserved.
// Authors: Chanwoo Lee
// Contacts: leechanwoo25@hanyang.ac.kr

#ifndef CW_CLOUD__CLOUD_VISUALIZER_HPP_
#define CW_CLOUD__CLOUD_VISUALIZER_HPP_

#include <pcl/impl/point_types.hpp>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/common/angles.h> // for pcl::deg2rad
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

#include <spdlog/common.h>
#include <spdlog/spdlog.h>

#include <memory>
#include <chrono>
#include <thread>

using namespace std::chrono_literals;

namespace cwcloud {

class CloudVisualizer
{
public:
  CloudVisualizer(
    const std::string & root_dir);
  CloudVisualizer()=delete;
  ~CloudVisualizer() {}

  void setCloud(
    pcl::PointCloud<pcl::PointXYZI> cloud,
    const std::string & cloud_id);

  void run();

private:
  pcl::visualization::PCLVisualizer::Ptr viewer_;
  std::string root_dir_;
};

} // namespace cwcloud

#endif
