// My Point Cloud Tools
#ifndef CXX_TOOLS__PCPR_HPP_
#define CXX_TOOLS__PCPR_HPP_
#pragma once

#include <cxxutil/CSVRow.hpp>

#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <spdlog/common.h>
#include <spdlog/fmt/bundled/core.h>
#include <spdlog/fmt/bundled/format.h>
#include <spdlog/spdlog.h>

#include <string>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;

namespace pcpr {

auto vec3dToPtXyz(
  const Eigen::Vector3d & vec)
  -> pcl::PointXYZ;
auto vec3dToPtXyzi(
  const Eigen::Vector3d & vec)
  -> pcl::PointXYZI;

auto ptXyziToVec3d(
  const pcl::PointXYZI &pi)
  -> Eigen::Vector3d;
auto ptXyzToVec3d(
  const pcl::PointXYZ &pi)
  -> Eigen::Vector3d;

auto strip(
  const std::string & str)
  -> std::string;

struct MetaPerScan
{
public:
  MetaPerScan()=delete;
  MetaPerScan(
    const std::string & _path,
    const double & _northing,
    const double & _easting)
    : path(_path),
      northing(_northing),
      easting(_easting) {}
  std::string path;
  double northing;
  double easting;
};

class NearestMeta
{
public:
  NearestMeta()=default;
  ~NearestMeta()=default;
  size_t size() const { return top_k_indices.size(); }
  std::vector<size_t> top_k_indices;
  std::vector<double> top_k_dists;
};

using MetaVec = std::vector<MetaPerScan>;
using DBaseCatalog = std::vector<std::vector<MetaPerScan>>;
using QueryCatalog = std::vector<MetaPerScan>;

auto readNearestMetas(
  const fs::path & data_fn)
  -> std::vector<NearestMeta>;

auto readCatalog(
  const fs::path & catalog_dir)
  -> std::tuple<DBaseCatalog, QueryCatalog>;

auto readScanList(
  const fs::path & list_fn,
  const double & skip_len=-1.0)
  -> MetaVec;

/**
 * @brief read boreas dataset lidar_poses.csv format
 * => GPSTime,easting,northing,altitude,vel_east,vel_north,vel_up,roll,pitch,heading,angvel_z,angvel_y,angvel_x
 * Skip 1 header.
 * Each row has 13 elements.
 */
auto read_scan_list_boreas(
  const fs::path & list_fn,
  const double & skip_len)
  -> MetaVec;

auto read_boreas_scan(
  const std::string & scan_fn,
  const double & ground_z,
  const double & sphere)
  ->pcl::PointCloud<pcl::PointXYZ>;

auto readCloudXyz64(
  const std::string & scan_fn)
  -> pcl::PointCloud<pcl::PointXYZ>;

void writeCloudXyz64(
  const std::string & scan_fn,
  const pcl::PointCloud<pcl::PointXYZ> & cloud);

void write_cloud_xyzi_float(
  const std::string & scan_fn,
  const pcl::PointCloud<pcl::PointXYZI> & cloud);

auto scalingCloud(
  const pcl::PointCloud<pcl::PointXYZ> & cloud,
  const double & scale)
  -> pcl::PointCloud<pcl::PointXYZ>;

}; // namespace pcpr

#endif
