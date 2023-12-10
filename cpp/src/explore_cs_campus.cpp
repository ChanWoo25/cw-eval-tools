#include <cmath>
#include <cstdlib>
#include <limits>
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
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>

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

struct isSpace
{
  bool operator()(unsigned c)
  {
    return (c == ' ' || c == '\n' || c == '\r' ||
            c == '\t' || c == '\v' || c == '\f');
  }
};

class CSVRow
{
private:
  std::string         line_;
  std::vector<int>    data_;

public:
  inline std::size_t size() const { return data_.size() - 1; }

  std::string_view operator[](std::size_t index) const
  {
    return std::string_view(&line_[data_[index] + 1], data_[index + 1] -  (data_[index] + 1));
  }

  void readNextRow(std::istream & str)
  {
    do {
      std::getline(str, line_);
    } while (line_[0] == '#');

    data_.clear();
    data_.emplace_back(-1);
    std::string::size_type pos = 0;

    line_.erase(std::remove_if(line_.begin(), line_.end(), isSpace()), line_.end());

    while((pos = line_.find(',', pos)) != std::string::npos)
    {
      data_.emplace_back(pos);
      ++pos;
    }

    // This checks for a trailing comma with no data after it.
    pos   = line_.size();
    data_.emplace_back(pos);
  }
};

std::istream & operator >> (
  std::istream & str,
  CSVRow & data)
{
  data.readNextRow(str);
  return str;
}

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

struct Config
{
  int target_n_points = 4096;
  std::string root_dir;
  int database_index {-1};
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

using MetaVec = std::vector<MetaPerScan>;
auto readScanList(
  const fs::path & list_fn,
  const double & skip_len)
  ->MetaVec
{
  MetaVec meta_vec;
  CSVRow csv_row;
  constexpr size_t N_COL = 3UL;

  double x_min = std::numeric_limits<double>::max();
  double x_max = std::numeric_limits<double>::min();
  double y_min = std::numeric_limits<double>::max();
  double y_max = std::numeric_limits<double>::min();
  double total_len = 0.0;
  size_t total_line = 0UL;
  size_t skiped_line = 0UL;

  std::ifstream fin(list_fn);
  fin >> csv_row; // Throw header
  while (true)    // Main
  {
    fin >> csv_row;
    if (csv_row.size() < N_COL) { break; }
    ++total_line;
    const auto path = std::string(csv_row[0]);
    const auto northing = std::stof(std::string(csv_row[1]));
    const auto easting  = std::stof(std::string(csv_row[2]));

    if (skip_len > 0.0 && !meta_vec.empty())
    {
      const auto dist2back = std::sqrt(
          std::pow(meta_vec.back().northing - northing, 2.0)
        + std::pow(meta_vec.back().easting  - easting, 2.0));
      if (dist2back < skip_len) { continue; }
    }

    x_min = (northing < x_min) ? (northing) : (x_min);
    x_max = (northing > x_max) ? (northing) : (x_max);
    y_min = (easting < y_min) ? (easting) : (y_min);
    y_max = (easting > y_max) ? (easting) : (y_max);
    if (!meta_vec.empty())
    {
      total_len += std::sqrt(
          std::pow(meta_vec.back().northing - northing, 2.0)
        + std::pow(meta_vec.back().easting  - easting, 2.0));
    }
    meta_vec.emplace_back(path, northing, easting);
    ++skiped_line;
  }
  fin.close();
  spdlog::info("x range: {:.3f} ~ {:.3f}", x_min, x_max);
  spdlog::info("y range: {:.3f} ~ {:.3f}", y_min, y_max);
  spdlog::info("total_length: {:.3f}", total_len);
  spdlog::info("total_line: {}", total_line);
  spdlog::info("skiped_line: {}", skiped_line);
  return meta_vec;
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

auto main(int argc, char * argv[]) -> int32_t
{
  if (argc < 3) {
    spdlog::error("Not enough arguments.");
    spdlog::error("[Usage] explore_cs_campus [aerial/ground] [seq_num]");
  }

  const auto type = std::string(argv[1]);
  const auto seq_num = std::stoi(std::string(argv[2]));
  if (type != "aerial" && type != "ground")
  {
    spdlog::error("Unknown type {}", type);
  }
  else if (   type == "aerial"
           && !(0 <= seq_num && seq_num < 2))
  {
    spdlog::error("Unknown aerial seq: {}", seq_num);
  }
  else if (   type == "ground"
           && !(1 <= seq_num && seq_num < 20))
  {
    spdlog::error("Unknown aerial seq: {}", seq_num);
  }

  const auto skip_len
    = (argc == 4)
    ? (std::stof(std::string(argv[3])))
    : -1.0;

  fs::path datasets_dir("/data/datasets");
  const auto cs_campus_dir = datasets_dir / "dataset_cs_campus";
  const auto benchmark_dir = cs_campus_dir / "benchmark_datasets";
  const auto umd_dir = benchmark_dir / "umd";
  const auto debug_dir = cs_campus_dir / "debug";
  const auto catalog_dir = cs_campus_dir / "catalog";
  const auto selected_dir
    = (type == "aerial")
    ? (umd_dir / fmt::format("umcp_lidar5_cloud_{}", (seq_num+6)))
    : (umd_dir / fmt::format("umcp_lidar5_ground_umd_gps_{}", seq_num));
  const auto list_fn = selected_dir / "umd_aerial_cloud_20m_100coverage_4096.csv";

  spdlog::info("Read {}.", list_fn.string());
  const auto meta_list = readScanList(list_fn, skip_len);

  /* Initialize Visualizer */
  cwcloud::CloudVisualizer vis("explore_cs_campus", 1, 1);
  const auto IDX_MIN = 0UL;
  const auto IDX_MAX = meta_list.size()-1UL;
  size_t idx = 0UL;

  while (!vis.wasStopped())
  {
    { /*  Register Query Cloud  & Clean other grids */
      const auto scan_fn = benchmark_dir / meta_list[idx].path;
      const auto scan = readCloudXYZ64(scan_fn.string());
      // spdlog::info("Scan path '{}' | n_points: {}", scan_fn.string(), scan.size());
      spdlog::info(
        "[{:04d}-Scan] north({:.4f}), east({:.4f})",
        idx,
        meta_list[idx].northing,
        meta_list[idx].easting);
      vis.setCloudXYZ(scan, 0, 0);
    }

    vis.run();

    if (vis.getKeySym() == "Right" && idx != IDX_MAX) { ++idx; }
    else if (vis.getKeySym() == "Left" && idx != IDX_MIN) { --idx; }
  }

  return EXIT_SUCCESS;
}
