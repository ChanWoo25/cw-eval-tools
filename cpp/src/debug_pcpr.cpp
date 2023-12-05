// #include <opencv2/opencv.hpp>
// #include <cv_bridge/cv_bridge.h>

#include <memory>
#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/sample_consensus/method_types.h>
#include <spdlog/common.h>
#include <spdlog/fmt/bundled/format.h>
#include <spdlog/spdlog.h>


#include <Eigen/Dense>
#include <Eigen/Geometry>
// #include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
// #include <pcl/filters/voxel_grid.h>
// #include <sensor_msgs/PointCloud2.h>
// #include <livox_ros_driver2/CustomMsg.h>

#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <tuple>
#include <filesystem>
#include <random>

#include "cwcloud/CloudVisualizer.hpp"

namespace fs = std::filesystem;

// std::vector<double> timestamps;
// std::vector<Eigen::Matrix3d> rotations;
// std::vector<Eigen::Quaterniond> quaternions;
// std::vector<Eigen::Vector3d> translations;
// std::vector<std::string> scan_fns;
// size_t N_SCAN = 0UL;

// void readPoses(const std::string & seq_dir)
// {
//   auto data_fn = fmt::format("{}/pose_fast_lio2.txt", seq_dir);
//   std::fstream f_data(data_fn.c_str(), std::ios::in);
//   // ROS_ASSERT(f_data.is_open());
//   std::string line;
//   while (std::getline(f_data, line))
//   {
//     if (line.empty()) { break; }
//     std::istringstream iss(line);
//     double timestamp;
//     Eigen::Quaterniond quat;
//     Eigen::Vector3d tran;
//     iss >> timestamp;
//     iss >> tran(0) >> tran(1) >> tran(2);
//     iss >> quat.x() >> quat.y() >> quat.z() >> quat.w();
//     timestamps.push_back(timestamp);
//     quaternions.push_back(quat);
//     rotations.push_back(quat.toRotationMatrix());
//     translations.push_back(tran);
//   }
//   f_data.close();

//   // ROS_ASSERT(timestamps.size() != 0UL);
//   // ROS_ASSERT(timestamps.size() == translations.size());
//   // ROS_ASSERT(timestamps.size() == quaternions.size());
//   // ROS_ASSERT(timestamps.size() == rotations.size());
//   N_SCAN = timestamps.size();
//   fmt::print("[readPoses] N_SCAN (updated): {}\n", N_SCAN);
// }

// void readLidarData(const std::string & seq_dir)
// {
//   auto data_fn = fmt::format("{}/lidar_data.txt", seq_dir);
//   std::fstream f_data(data_fn.c_str(), std::ios::in);
//   std::string line;
//   while (std::getline(f_data, line))
//   {
//     if (line.empty()) { break; }
//     std::stringstream ss;
//     ss << line;
//     double timestamp;
//     std::string scan_fn;

//     ss >> timestamp >> scan_fn;
//     scan_fns.push_back(fmt::format("{}/{}", seq_dir, scan_fn));
//     if (scan_fns.size() == timestamps.size()) { break; }
//   }
//   f_data.close();

//   // ROS_ASSERT(scan_fns.size() == timestamps.size());
//   fmt::print("[readLidarData] 1st Scan '{}'\n", scan_fns.front());
//   fmt::print("[readLidarData] Fin Scan '{}'\n", scan_fns.back());
// }

// pcl::PointXYZI vec2point(const Eigen::Vector3d &vec) {
//   pcl::PointXYZI pi;
//   pi.x = vec[0];
//   pi.y = vec[1];
//   pi.z = vec[2];
//   return pi;
// }
// pcl::PointXYZ vec2pt_XYZ(const Eigen::Vector3d &vec) {
//   pcl::PointXYZ pt;
//   pt.x = vec[0];
//   pt.y = vec[1];
//   pt.z = vec[2];
//   return pt;
// }

// Eigen::Vector3d pt2vec(const pcl::PointXYZI &pi) {
//   return Eigen::Vector3d(pi.x, pi.y, pi.z);
// }
// Eigen::Vector3d pt2vec_XYZ(const pcl::PointXYZ &pi) {
//   return Eigen::Vector3d(pi.x, pi.y, pi.z);
// }

// auto readCloudXYZ64(
//   const std::string & scan_fn)
//   ->pcl::PointCloud<pcl::PointXYZ>
// {
//   std::fstream f_bin(scan_fn, std::ios::in | std::ios::binary);
//   // ROS_ASSERT(f_bin.is_open());

//   f_bin.seekg(0, std::ios::end);
//   const size_t num_elements = f_bin.tellg() / sizeof(double);
//   std::vector<double> buf(num_elements);
//   f_bin.seekg(0, std::ios::beg);
//   f_bin.read(
//     reinterpret_cast<char *>(
//       &buf[0]),
//       num_elements * sizeof(double));
//   f_bin.close();
//   // fmt::print("Add {} pts\n", num_elements/4);
//   pcl::PointCloud<pcl::PointXYZ> cloud;
//   for (std::size_t i = 0; i < buf.size(); i += 3)
//   {
//     pcl::PointXYZ point;
//     point.x = buf[i];
//     point.y = buf[i + 1];
//     point.z = buf[i + 2];
//     Eigen::Vector3d pv = pt2vec_XYZ(point);
//     point = vec2pt_XYZ(pv);
//     cloud.push_back(point);
//   }
//   return cloud;
// }

// auto getCloudMeanStd(
//   const pcl::PointCloud<pcl::PointXYZI> & cloud)
//   ->std::tuple<Eigen::Vector3f, float>
// {
//   if (cloud.points.empty()) { return {Eigen::Vector3f::Zero(), 0.0f}; }
//   const auto n_pts = static_cast<float>(cloud.size());
//   Eigen::Vector3f center = Eigen::Vector3f::Zero();
//   for (const auto & point: cloud.points)
//   {
//     center(0) += point.x;
//     center(1) += point.y;
//     center(2) += point.z;
//   }
//   center /= n_pts;
//   auto stddev = 0.0f;
//   for (const auto & point: cloud.points)
//   {
//     stddev += std::sqrt(
//       std::pow(point.x - center(0), 2.0f) +
//       std::pow(point.y - center(1), 2.0f) +
//       std::pow(point.z - center(2), 2.0f));
//   }
//   stddev /= n_pts;
//   return {center, stddev};
// }

// void writeScanBinary(
//   const std::string & scan_fn,
//   const pcl::PointCloud<pcl::PointXYZI> & cloud)
// {
//   std::ofstream f_scan(scan_fn, std::ios::out | std::ios::binary);
//   // ROS_ASSERT(f_scan.is_open());
//   for (const auto & point : cloud.points)
//   {
//     f_scan.write(reinterpret_cast<const char *>(&point.x), sizeof(float));
//     f_scan.write(reinterpret_cast<const char *>(&point.y), sizeof(float));
//     f_scan.write(reinterpret_cast<const char *>(&point.z), sizeof(float));
//     f_scan.write(reinterpret_cast<const char *>(&point.intensity), sizeof(float));
//   }
//   f_scan.close();
// }

// void processSequence(
//   const std::string & seq_dir,
//   const size_t & target_num_points)
// {
//   N_SCAN = 0UL;
//   timestamps.clear();
//   translations.clear();
//   quaternions.clear();
//   rotations.clear();
//   scan_fns.clear();
//   readPoses(seq_dir);
//   readLidarData(seq_dir);

//   auto keyframe_data_fn = fmt::format("{}/lidar_keyframe_data.txt", seq_dir);
//   auto keyframe_data_dir = fmt::format("{}/lidar_keyframe_downsampled", seq_dir);
//   fs::create_directories(keyframe_data_dir);
//   std::fstream f_key_data (keyframe_data_fn, std::ios::out);

//   size_t prev_idx {10UL};
//   size_t curr_idx {prev_idx};
//   // Eigen::Vector3d prev_pos = Eigen::Vector3d::Zero();
//   // Eigen::Vector3d curr_pos = Eigen::Vector3d::Zero();
//   for (curr_idx = prev_idx; curr_idx < N_SCAN; ++curr_idx)
//   {
//     if (prev_idx == curr_idx)
//     {
//       auto & tran = translations[curr_idx];
//       auto & quat = quaternions[curr_idx];
//       auto line = fmt::format(
//         "{:.6f} lidar_keyframe_downsampled/{:06d}.bin {:6f} {:6f} {:6f} {:6f} {:6f} {:6f} {:6f}\n",
//         timestamps[curr_idx], curr_idx,
//         tran(0), tran(1), tran(2),
//         quat.x(), quat.y(), quat.z(), quat.w());
//       f_key_data << line;
//       /* Downsample cloud into target number of points */
//       auto cloud = readCloud(scan_fns[curr_idx]);
//       auto downsampled = downsamplingUntilTargetNumPoints(cloud, target_num_points);
//       auto normalized = normalizeCloud(downsampled);
//       auto new_scan_fn = fmt::format("{}/{:06d}.bin", keyframe_data_dir, curr_idx);
//       writeScanBinary(new_scan_fn, normalized);
//       fmt::print("[processSequence] Write to {}\n", new_scan_fn);
//     }
//     else
//     {
//       auto & prev_t = translations[prev_idx];
//       auto & curr_t = translations[curr_idx];
//       auto dist = (curr_t - prev_t).norm();
//       auto & prev_R = rotations[prev_idx];
//       auto & curr_R = rotations[curr_idx];
//       Eigen::Matrix3d dist_R = prev_R.transpose() * curr_R;
//       Eigen::Vector3d dist_r = Eigen::AngleAxisd(dist_R).angle() * Eigen::AngleAxisd(dist_R).axis();
//       if (dist >= 0.5
//           || dist_r(0) >= (M_PI * (10.0/180.0))
//           || dist_r(1) >= (M_PI * (10.0/180.0))
//           || dist_r(2) >= (M_PI * (10.0/180.0)))
//       {
//         // fmt::print(
//         //   "dist({:.3f}), droll({:.3f}), dpitch({:.3f}), dyaw({:.3f})\n",
//         //   dist, dist_r(0), dist_r(1), dist_r(2));

//         auto & tran = translations[curr_idx];
//         auto & quat = quaternions[curr_idx];
//         auto line = fmt::format(
//           "{:.6f} lidar_keyframe_downsampled/{:06d}.bin {:6f} {:6f} {:6f} {:6f} {:6f} {:6f} {:6f}\n",
//           timestamps[curr_idx], curr_idx,
//           tran(0), tran(1), tran(2),
//           quat.x(), quat.y(), quat.z(), quat.w());
//         f_key_data << line;
//         /* Downsample cloud into target number of points */
//         auto cloud = readCloud(scan_fns[curr_idx]);
//         auto downsampled = downsamplingUntilTargetNumPoints(cloud, target_num_points);
//         auto normalized = normalizeCloud(downsampled);
//         auto new_scan_fn = fmt::format("{}/{:06d}.bin", keyframe_data_dir, curr_idx);
//         writeScanBinary(new_scan_fn, normalized);
//         fmt::print("[processSequence] Write to {}\n", new_scan_fn);
//         prev_idx = curr_idx;
//       }
//     }
//   }
//   f_key_data.close();
// }



struct Config
{
  int target_n_points = 4096;
  std::string root_dir;
} cfg ;

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

using DBaseCatalog = std::vector<std::vector<MetaPerScan>>;
using QueryCatalog = std::vector<MetaPerScan>;
auto readCatalog(
  const fs::path & catalog_dir)
  -> std::tuple<DBaseCatalog, QueryCatalog>
{
  DBaseCatalog dbase_catalog;
  QueryCatalog query_catalog;
  dbase_catalog.resize(14, std::vector<MetaPerScan>());

  /* Read 14 Database catalogs */
  for (uint32_t i = 0U; i < 14; ++i)
  {
    const auto catalog_fn = catalog_dir / fmt::format("db_catalog_{}.txt", i);
    spdlog::info("Read {} ...", catalog_fn.string());
    std::ifstream fin(catalog_fn.string());
    std::string line;
    int n_lines = 0;
    while (std::getline(fin, line))
    {
      if (line.empty()) { break; }
      ++n_lines;
      std::istringstream iss(line);
      int index;
      std::string bin_path;
      double northing;
      double easting;

      iss >> index >> bin_path >> northing >> easting;
      dbase_catalog[i].emplace_back(bin_path, northing, easting);
    }
    fin.close();
    spdlog::info("{:02d}-th db size: {}", i, n_lines);
  }

  /* Read Query catalogs */
  const auto catalog_fn = catalog_dir / fmt::format("qr_catalog.txt");
  spdlog::info("Read {} ...", catalog_fn.string());
  std::ifstream fin(catalog_fn.string());
  std::string line;
  int n_lines = 0;
  while (std::getline(fin, line))
  {
    if (line.empty()) { break; }
    ++n_lines;
    std::istringstream iss(line);
    int index;
    std::string bin_path;
    double northing;
    double easting;

    iss >> index >> bin_path >> northing >> easting;
    query_catalog.emplace_back(bin_path, northing, easting);
  }
  fin.close();
  spdlog::info("qr size: {}", n_lines);

  return std::make_tuple(dbase_catalog, query_catalog);
}

auto main() -> int32_t
{
  fs::path datasets_dir("/data/datasets");
  const auto cs_campus_dir = datasets_dir / "dataset_cs_campus";
  const auto benchmark_dir = cs_campus_dir / "benchmark_datasets";
  const auto umd_dir = benchmark_dir / "umd";
  const auto debug_dir = cs_campus_dir / "debug";
  const auto catalog_dir = cs_campus_dir / "catalog";

  // auto save_dir = cs_campus_dir;
  const auto save_fn = cs_campus_dir / "test.txt";
  spdlog::info("save_fn: {}", save_fn.string());
  // fs::create_directory(save_dir);
  // spdlog::info("current_path: {}", fs::current_path().string());
  // spdlog::info("relative_path: {}", root_dir.relative_path().string());
  // spdlog::info("absolute_path: {}", fs::absolute(root_dir).string());
  // spdlog::info("canonical_path: {}", fs::canonical(root_dir).string());

  const auto [dbase_catalog, query_catalog]
    = readCatalog(catalog_dir);

  std::ofstream fout(save_fn); int idx = 0;
  for (const auto & catalog: dbase_catalog)
  {
    fout << fmt::format("{}-th db, length: {}", idx, catalog.size());
  }
  fout << fmt::format("qeuries, length: {}", query_catalog.size());
  fout.close();

  cwcloud::CloudVisualizer vis("Debug", 3, 3);

  return EXIT_SUCCESS;
}
