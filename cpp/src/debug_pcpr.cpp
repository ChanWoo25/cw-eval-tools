// #include <opencv2/opencv.hpp>
// #include <cv_bridge/cv_bridge.h>

#include <cmath>
#include <memory>
#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/sample_consensus/method_types.h>
#include <spdlog/common.h>
#include <spdlog/fmt/bundled/core.h>
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

std::string strip(const std::string & str)
{
    auto start_it = str.begin();
    auto end_it = str.rbegin();
    while (std::isspace(*start_it)) { ++start_it; }
    while ( std::isspace(*end_it) ) {  ++end_it;  }
    const auto len = end_it.base() - start_it;
    return (len <= 0)
           ? std::string("")
           : std::string(start_it, end_it.base());
}

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

pcl::PointXYZI vec2point(const Eigen::Vector3d &vec) {
  pcl::PointXYZI pi;
  pi.x = vec[0];
  pi.y = vec[1];
  pi.z = vec[2];
  return pi;
}

pcl::PointXYZ vec2pt_XYZ(const Eigen::Vector3d &vec) {
  pcl::PointXYZ pt;
  pt.x = vec[0];
  pt.y = vec[1];
  pt.z = vec[2];
  return pt;
}

Eigen::Vector3d pt2vec(const pcl::PointXYZI &pi) {
  return Eigen::Vector3d(pi.x, pi.y, pi.z);
}
Eigen::Vector3d pt2vec_XYZ(const pcl::PointXYZ &pi) {
  return Eigen::Vector3d(pi.x, pi.y, pi.z);
}

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
  }

  /* Read Query catalogs */
  const auto catalog_fn = catalog_dir / fmt::format("qr_catalog.txt");
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

  return std::make_tuple(dbase_catalog, query_catalog);
}

using StateVec = std::vector<std::string>;
using PosVec = std::vector<int>;
using NegVec = std::vector<std::vector<int>>;
using ScoreVec = std::vector<double>;
using ScoreVec = std::vector<double>;
auto readDebugFile(
  const fs::path & debug_fn)
  -> std::tuple<StateVec, PosVec, NegVec, ScoreVec>
{
  std::ifstream fin(debug_fn);
  StateVec state_vec;
  PosVec pos_vec;
  NegVec neg_vec;
  ScoreVec score_vec;

  std::string line;
  int n_lines = 0;
  unsigned cnt_found = 0;
  unsigned cnt_not_found = 0;
  unsigned cnt_no_answer = 0;
  while (std::getline(fin, line))
  {
    line = strip(line);
    if (line.empty()) { break; }
    ++n_lines;
    state_vec.push_back("Fail");
    pos_vec.push_back(-1);
    neg_vec.push_back(std::vector<int>());
    score_vec.push_back(-1.0);

    std::istringstream iss(line);
    int query_index;
    std::string state;
    int match_index;
    double score;
    iss >> query_index;

    char test_char;
    while (iss.readsome(&test_char, 1) != 0)
    {
      iss >> state >> match_index;
      if (state == "T")
      {
        iss >> score;
        state_vec.back() = "Find";
        pos_vec.back() = match_index;
        score_vec.back() = score;
        break;
      }
      else if (state == "F")
      {
        neg_vec.back().push_back(match_index);
      }
      else
      {
        break;
        spdlog::warn("Something Wrong, {}", state);
      }
    }

    if (state_vec.back() == "Fail" && neg_vec.back().empty()) {
      state_vec.back() = "None";
      ++cnt_no_answer;
    } else {
      if (state_vec.back() == "Find") {
        ++cnt_found;
      } else {
        ++cnt_not_found;
      }
    }

    if (n_lines >= 1059)
    {
      break;
    }
  }

  fin.close();
  spdlog::info("debug size: {} | Found({}), Not found({}), No answer({})",
    n_lines, cnt_found, cnt_not_found, cnt_no_answer);
  return std::make_tuple(state_vec, pos_vec, neg_vec, score_vec);
}

auto readCloudXYZ64(
  const std::string & scan_fn)
  ->pcl::PointCloud<pcl::PointXYZ>
{
  std::fstream f_bin(scan_fn, std::ios::in | std::ios::binary);
  // ROS_ASSERT(f_bin.is_open());

  f_bin.seekg(0, std::ios::end);
  const size_t num_elements = f_bin.tellg() / sizeof(double);
  std::vector<double> buf(num_elements);
  f_bin.seekg(0, std::ios::beg);
  f_bin.read(
    reinterpret_cast<char *>(
      &buf[0]),
      num_elements * sizeof(double));
  f_bin.close();
  // fmt::print("Add {} pts\n", num_elements/4);
  pcl::PointCloud<pcl::PointXYZ> cloud;
  for (std::size_t i = 0; i < buf.size(); i += 3)
  {
    pcl::PointXYZ point;
    point.x = buf[i];
    point.y = buf[i + 1];
    point.z = buf[i + 2];
    Eigen::Vector3d pv = pt2vec_XYZ(point);
    point = vec2pt_XYZ(pv);
    cloud.push_back(point);
  }
  return cloud;
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
  // const auto save_fn = cs_campus_dir / "test.txt";
  // spdlog::info("save_fn: {}", save_fn.string());
  // fs::create_directory(save_dir);
  // spdlog::info("current_path: {}", fs::current_path().string());
  // spdlog::info("relative_path: {}", root_dir.relative_path().string());
  // spdlog::info("absolute_path: {}", fs::absolute(root_dir).string());
  // spdlog::info("canonical_path: {}", fs::canonical(root_dir).string());

  spdlog::info("[readCatalog]");
  const auto [dbase_catalog, query_catalog]
    = readCatalog(catalog_dir);
  for (uint32_t di = 0U; di < dbase_catalog.size(); di++)
  {
    spdlog::info("- {}-th db, length: {}", di, dbase_catalog[di].size());
  }
  spdlog::info("- qeuries, length: {}", query_catalog.size());

  /* Let's see Histogram */
  auto getDistance
    = [](const double & x1, const double & y1,
         const double & x2, const double & y2) -> double
      {
        return std::sqrt(std::pow(x1-x2, 2.0) + std::pow(y1-y2, 2.0));
      };
  for (uint32_t di = 1U; di < dbase_catalog.size(); di++)
  {
    const auto & sub_dbase_catalog = dbase_catalog[di];

    /* Read corresponding debug file */
    const auto debug_fn = debug_dir / fmt::format("db-{}-qr-0-debug.txt", di);
    const auto output_fn = debug_dir / fmt::format("db-{}-qr-0-distances.txt", di);
    const auto [state_vec, pos_vec, neg_vec, score_vec]
      = readDebugFile(debug_fn);

    std::ofstream fout(output_fn);
    for (uint32_t qi = 0U; qi < query_catalog.size(); qi++)
    {
      fout << fmt::format("{:04d}", qi);
      const auto q_north = query_catalog[qi].northing;
      const auto q_east = query_catalog[qi].easting;

      if (state_vec[qi] == "Find")
      {
        const auto mi = pos_vec[qi];
        const auto m_north = sub_dbase_catalog[mi].northing;
        const auto m_east  = sub_dbase_catalog[mi].easting;
        const auto dist = getDistance(q_north, q_east, m_north, m_east);
        fout << fmt::format(" {} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n",
          state_vec[qi], q_north, q_east, m_north, m_east, dist, score_vec[qi]);
      }
      else if (state_vec[qi] == "Fail")
      {
        const auto mi = neg_vec[qi].front();
        const auto m_north = sub_dbase_catalog[mi].northing;
        const auto m_east  = sub_dbase_catalog[mi].easting;
        const auto dist = getDistance(q_north, q_east, m_north, m_east);
        fout << fmt::format(" {} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n",
          state_vec[qi], q_north, q_east, m_north, m_east, dist, score_vec[qi]);
      }
      else if (state_vec[qi] == "None")
      {
        fout << fmt::format(" {}\n", state_vec[qi]);
      }
      else
      {
        spdlog::warn("Something wrong.");
      }
    }
  }


  cwcloud::CloudVisualizer vis("debug_pcpr", 2, 3);

  /* Query */
  int qi = 19;
  {
    const auto scan_fn = benchmark_dir / query_catalog[qi].path;
    const auto scan = readCloudXYZ64(scan_fn.string());
    const auto row = 0;
    const auto col = 0;
    spdlog::info("read scan from {} | n_points: {}", scan_fn.string(), scan.size());
    vis.setCloudXYZ(scan, row, col);
  }

  int di = 1;
  const auto debug_fn = debug_dir / fmt::format("db-{}-qr-0-debug.txt", di);
  const auto output_fn = debug_dir / fmt::format("db-{}-qr-0-distances.txt", di);
  const auto [state_vec, pos_vec, neg_vec, score_vec]
    = readDebugFile(debug_fn);

  int ti = pos_vec[qi];
  {
    const auto scan_fn = benchmark_dir / dbase_catalog[di][ti].path;
    const auto scan = readCloudXYZ64(scan_fn.string());
    const auto row = 0;
    const auto col = 1;
    spdlog::info("read scan from {} | n_points: {}", scan_fn.string(), scan.size());
    vis.setCloudXYZ(scan, row, col);
  }

  {
    int fi = neg_vec[qi][0];
    const auto scan_fn = benchmark_dir / dbase_catalog[di][fi].path;
    const auto scan = readCloudXYZ64(scan_fn.string());
    const auto row = 0;
    const auto col = 2;
    spdlog::info("read scan from {} | n_points: {}", scan_fn.string(), scan.size());
    vis.setCloudXYZ(scan, row, col);
  }
  {
    int fi = neg_vec[qi][1];
    const auto scan_fn = benchmark_dir / dbase_catalog[di][fi].path;
    const auto scan = readCloudXYZ64(scan_fn.string());
    const auto row = 1;
    const auto col = 0;
    spdlog::info("read scan from {} | n_points: {}", scan_fn.string(), scan.size());
    vis.setCloudXYZ(scan, row, col);
  }
  {
    int fi = neg_vec[qi][2];
    const auto scan_fn = benchmark_dir / dbase_catalog[di][fi].path;
    const auto scan = readCloudXYZ64(scan_fn.string());
    const auto row = 1;
    const auto col = 1;
    spdlog::info("read scan from {} | n_points: {}", scan_fn.string(), scan.size());
    vis.setCloudXYZ(scan, row, col);
  }
  {
    int fi = neg_vec[qi][3];
    const auto scan_fn = benchmark_dir / dbase_catalog[di][fi].path;
    const auto scan = readCloudXYZ64(scan_fn.string());
    const auto row = 1;
    const auto col = 2;
    spdlog::info("read scan from {} | n_points: {}", scan_fn.string(), scan.size());
    vis.setCloudXYZ(scan, row, col);
  }

  vis.run();

  return EXIT_SUCCESS;
}
