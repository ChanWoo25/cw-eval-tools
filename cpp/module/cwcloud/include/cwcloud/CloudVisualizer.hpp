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
#include <string>
#include <thread>
#include <unordered_map>

using namespace std::chrono_literals;

namespace cwcloud {

class CloudVisualizer
{
public:
  CloudVisualizer(
    const std::string & root_dir,
    const int nrows=1,
    const int ncols=1);
  CloudVisualizer()=delete;
  ~CloudVisualizer() {}

  void setCloud(
    pcl::PointCloud<pcl::PointXYZI> cloud,
    const std::string & cloud_id = "cloud",
    const int vid=0);

  void setCloudXYZ(
    pcl::PointCloud<pcl::PointXYZ> cloud,
    const int & nrow,
    const int & ncol,
    const std::string & _cloud_id="");

  void setGroundInformCloud(
    pcl::PointCloud<pcl::PointXYZI> cloud_ground,
    pcl::PointCloud<pcl::PointXYZI> cloud_inform,
    const std::string & cloud_id = "default-gi",
    const int vid=0);

  void run();

  inline
  auto getKeySym() const -> std::string { return key_sym_; }

private:
  pcl::visualization::PCLVisualizer::Ptr viewer_;
  std::string root_dir_;
  std::unordered_map<std::string, bool> cloud_ids_;
  bool skip_flag = false;
  std::vector<int> viewport_ids_;
  int nrows_, ncols_;
  std::string key_sym_;
};

} // namespace cwcloud

#endif
