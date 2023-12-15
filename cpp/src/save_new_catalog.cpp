#include <cfloat>
#include <cmath>
#include <cstddef>
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
#include <argparse/argparse.hpp>

#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <tuple>
#include <filesystem>
#include <random>

#include "cwcloud/CloudVisualizer.hpp"

namespace fs = std::filesystem;

fs::path datasets_dir("/data/datasets");
const auto cs_campus_dir = datasets_dir / "dataset_cs_campus";
const auto benchmark_dir = cs_campus_dir / "benchmark_datasets";
const auto umd_dir = benchmark_dir / "umd";
const auto debug_dir = cs_campus_dir / "debug";

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

class Config
{
public:
  Config() {
    test_sectors[0][0] = 4317250.0; test_sectors[0][1] = 331950.0;
    test_sectors[1][0] = 4317290.0; test_sectors[1][1] = 332130.0;
    test_sectors[2][0] = 4317390.0; test_sectors[2][1] = 332260.0;
    test_sectors[3][0] = 4317470.0; test_sectors[3][1] = 331930.0;
    test_sectors[4][0] = 4317480.0; test_sectors[4][1] = 332100.0;
    test_sectors[5][0] = 4317520.0; test_sectors[5][1] = 332210.0;

    aerial_sequences[0] = "umcp_lidar5_cloud_6";
    aerial_sequences[1] = "umcp_lidar5_cloud_7";

    ground_sequences[0] = "umcp_lidar5_ground_umd_gps_1";
    ground_sequences[1] = "umcp_lidar5_ground_umd_gps_2";
    ground_sequences[2] = "umcp_lidar5_ground_umd_gps_3";
    ground_sequences[3] = "umcp_lidar5_ground_umd_gps_4";
    ground_sequences[4] = "umcp_lidar5_ground_umd_gps_5";
    ground_sequences[5] = "umcp_lidar5_ground_umd_gps_6";
    ground_sequences[6] = "umcp_lidar5_ground_umd_gps_7";
    ground_sequences[7] = "umcp_lidar5_ground_umd_gps_8";
    ground_sequences[8] = "umcp_lidar5_ground_umd_gps_9";
    ground_sequences[9] = "umcp_lidar5_ground_umd_gps_10";
    ground_sequences[10] = "umcp_lidar5_ground_umd_gps_11";
    ground_sequences[11] = "umcp_lidar5_ground_umd_gps_12";
    ground_sequences[12] = "umcp_lidar5_ground_umd_gps_13";
    ground_sequences[13] = "umcp_lidar5_ground_umd_gps_14";
    ground_sequences[14] = "umcp_lidar5_ground_umd_gps_15";
    ground_sequences[15] = "umcp_lidar5_ground_umd_gps_16";
    ground_sequences[16] = "umcp_lidar5_ground_umd_gps_17";
    ground_sequences[17] = "umcp_lidar5_ground_umd_gps_18";
    ground_sequences[18] = "umcp_lidar5_ground_umd_gps_19";
  }
  double aerial_x_min = 4316500.0;
  double aerial_x_max = 4318500.0;
  double aerial_y_min = 331400.0;
  double aerial_y_max = 332600.0;

  double aerial_width = 50.0;
  double aerial_height = 50.0;
  double ground_width = 40.0;
  double ground_height = 40.0;
  using XY = std::array<double, 2>;
  std::array<XY, 6> test_sectors;

  std::array<std::string, 2> aerial_sequences;
  std::array<std::string, 19> ground_sequences;
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

// Define the point structure
struct Point {
  double x;
  double y;
};

// Euclidean distance function
double meta_dist(
  const MetaPerScan & p1,
  const MetaPerScan & p2) {
  return sqrt(  pow(p1.northing - p2.northing, 2)
              + pow(p1.easting  - p2.easting , 2));
}

/* Check that the (i)-th and (i+1)-th elements of aerial_meta have the same coordinates. */
void check_aerial_metas(const std::vector<MetaPerScan> & aerial_metas)
{
  for (size_t i = 0; i < aerial_metas.size(); i += 2)
  {
    const auto n_diff = std::abs(aerial_metas[i].northing - aerial_metas[i+1].northing);
    const auto e_diff = std::abs(aerial_metas[i].easting - aerial_metas[i+1].easting);
    if (n_diff > DBL_EPSILON || e_diff > DBL_EPSILON)
    {
      spdlog::warn("Diff with my assumption.");
      spdlog::warn("[{:04d}] {:10.4f}, {:10.4f}", i, aerial_metas[i].northing, aerial_metas[i].easting);
      spdlog::warn("[{:04d}] {:10.4f}, {:10.4f}", i, aerial_metas[i+1].northing, aerial_metas[i+1].easting);
    }
  }
}

auto find_top_k_neighbors(
  const MetaPerScan & ground,
  const std::vector<MetaPerScan> & aerial_metas,
  const int k)
  -> std::tuple<std::vector<size_t>, std::vector<double>>
{
  if (k < 1) { spdlog::error("k({}) < 1 not allowed.", k); exit(1); }

  std::vector<size_t> top_k_indices(k, 0UL);
  std::vector<double> top_k_dists  (k, std::numeric_limits<double>::max());
  for (size_t i = 0UL; i < aerial_metas.size(); i += 2UL)
  {
    const auto dist = meta_dist(ground, aerial_metas[i]);

    if (dist < top_k_dists[0])
    {
      for (int j = 0; j < k-1; ++j)
      {
        top_k_dists[k-1-j]   = top_k_dists[k-1-(j+1)];
        top_k_indices[k-1-j] = top_k_indices[k-1-(j+1)];
      }
      top_k_indices[0] = i;
      top_k_dists[0] = dist;
    }
  }
  return std::make_tuple(top_k_indices, top_k_dists);
}

void find_n_save_top_k_neighbors(
  const fs::path & save_fn,
  const std::vector<MetaPerScan> & ground_metas,
  const std::vector<MetaPerScan> & aerial_metas,
  const int k)
{
  spdlog::info ("save_fn: {}, top_k: {}", save_fn.string(), k);
  std::ofstream f_out(save_fn);
  for (size_t i = 0; i < ground_metas.size(); ++i)
  {
    const auto & meta = ground_metas[i];
    const auto [top_k_indices, top_k_dists]
      = find_top_k_neighbors(meta, aerial_metas, k);

    /* Format: {path} {northing} {easting} {k} {1-th index} {1-th dist} ... {k-th index} {k-th dist} */
    std::string line = fmt::format(
      "{} {:.6f} {:.6f} {}",
      meta.path, meta.northing, meta.easting, k);
    for (int j = 0; j < k; j++)
    {
      line = line + fmt::format(" {} {}", top_k_indices[j], top_k_dists[j]);
    }
    if (i != 0) { line = "\n" + line; }
    f_out << line;
  }
  f_out.close();
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

using MetaVec = std::vector<MetaPerScan>;
auto readScanList(
  const fs::path & list_fn,
  const double & skip_len=-1.0)
  ->MetaVec;

void main_create_ver2(const double & skip_len);

void main_nearest();

auto main(int argc, char * argv[]) -> int32_t
{
  argparse::ArgumentParser program("save_new_catalog");

  // git add subparser
  argparse::ArgumentParser add_command("ver2");
  add_command.add_description("Version2 Catalog");
  add_command.add_argument("skip_length")
    .help("Skip length for avoiding redundant scans")
    .scan<'g', double>();

  // save_new_catalog nearest
  argparse::ArgumentParser nearest_command("nearest");
  nearest_command.add_description("save nearest aerial scan per ground scan [No arguments needed]");

  program.add_subparser(add_command);
  program.add_subparser(nearest_command);

  try {
    program.parse_args(argc, argv);
  }
  catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return EXIT_FAILURE;
  }

  if (program.is_subcommand_used("nearest"))
  {
    main_nearest();
  }
  if (program.is_subcommand_used("ver2"))
  {
    const auto skip_len = add_command.get<double>("skip_length");
    spdlog::info("Arg | skip_length: {:.2f}", skip_len);
    main_create_ver2(skip_len);
  }

  return EXIT_SUCCESS;
}


void main_nearest()
{
  spdlog::info(" ==================");
  spdlog::info(" || main_nearest ||");
  spdlog::info(" ==================");

  const auto catalog_dir
    = cs_campus_dir / "catalog-nearest-aerial";
  fs::create_directory(catalog_dir);
  spdlog::info("Save to \"{}\"", catalog_dir.string());
  if (!fs::exists(catalog_dir))
  {
    spdlog::error("==> Directory doesn't exist!!");
    exit(EXIT_FAILURE);
  }
  else
  {
    spdlog::info("==> Dir [ok]");
  }

  std::array<MetaVec, 2>  aerial_vec;
  std::array<MetaVec, 19> ground_vec;

  /* Aerial */
  for (int i = 0; i < 2; i++)
  {
    const auto & aerial_seq = cfg.aerial_sequences[i];
    const auto list_fn
      = umd_dir / aerial_seq / "umd_aerial_cloud_20m_100coverage_4096.csv";
    spdlog::info("Read {}", list_fn.string());
    aerial_vec[i] = readScanList(list_fn);
    check_aerial_metas(aerial_vec[i]);
  }

  /* Ground */
  for (int i = 0; i < 19; i++)
  {
    const auto & groumd_seq = cfg.ground_sequences[i];
    const auto list_fn
      = umd_dir / groumd_seq / "umd_aerial_cloud_20m_100coverage_4096.csv";
    spdlog::info("Read {}", list_fn.string());
    ground_vec[i] = readScanList(list_fn);
  }

  for (size_t i = 0; i < aerial_vec.size(); ++i)
  {
    for (size_t j = 0 ; j < ground_vec.size(); ++j)
    {
      constexpr int top_k = 4;
      const auto save_fn
        = catalog_dir
          / fmt::format(
              "ground{:02d}-aerial{:02d}-top_{}-nearest.txt",
              j, i, top_k);
      find_n_save_top_k_neighbors(
        save_fn,
        ground_vec[j],
        aerial_vec[i],
        top_k);
    }
  }
}

void main_create_ver2(const double & skip_len)
{
  const auto catalog_dir = cs_campus_dir / "catalog";
  const auto catalog_ver2_dir = cs_campus_dir / "catalog_ver2";
  fs::create_directory(catalog_ver2_dir);

  MetaVec training_vec;
  std::array<MetaVec, 2>  aerial_database_vec;
  std::array<MetaVec, 19> ground_database_vec;
  MetaVec queries_vec;

  /* Aerial */
  for (int i = 0; i < 2; i++)
  {
    const auto & aerial_seq = cfg.aerial_sequences[i];
    const auto list_fn
      = umd_dir / aerial_seq / "umd_aerial_cloud_20m_100coverage_4096.csv";
    spdlog::info("Read {}", list_fn.string());
    const auto meta_list = readScanList(list_fn, skip_len);
    for (const auto & meta: meta_list)
    {
      if (   cfg.aerial_x_min <= meta.northing && meta.northing <= cfg.aerial_x_max
          && cfg.aerial_y_min <= meta.easting  && meta.easting  <= cfg.aerial_y_max)
      {
        bool is_training_set = true;
        for (const auto & test_sector: cfg.test_sectors)
        {
          if (   test_sector[0] - cfg.aerial_width < meta.northing && meta.northing < test_sector[0] + cfg.aerial_width
              && test_sector[1] - cfg.aerial_height < meta.easting && meta.easting < test_sector[1] + cfg.aerial_height)
          {
            is_training_set = false;
            break;
          }
        }

        aerial_database_vec[i].push_back(meta);
        if (is_training_set)
        {
          training_vec.push_back(meta);
        }
        else
        {
          queries_vec.push_back(meta);
        }
      }
    }
  }

  /* Ground */
  for (int i = 0; i < 19; i++)
  {
    const auto & groumd_seq = cfg.ground_sequences[i];
    const auto list_fn
      = umd_dir / groumd_seq / "umd_aerial_cloud_20m_100coverage_4096.csv";
    spdlog::info("Read {}", list_fn.string());
    const auto meta_list = readScanList(list_fn, skip_len);
    for (const auto & meta: meta_list)
    {
      bool is_training_set = true;
      for (const auto & test_sector: cfg.test_sectors)
      {
        if (   test_sector[0] - cfg.ground_width < meta.northing && meta.northing < test_sector[0] + cfg.ground_width
            && test_sector[1] - cfg.ground_height < meta.easting && meta.easting < test_sector[1] + cfg.ground_height)
        {
          is_training_set = false;
          break;
        }
      }

      ground_database_vec[i].push_back(meta);
      if (is_training_set)
      {
        training_vec.push_back(meta);
      }
      else
      {
        queries_vec.push_back(meta);
      }
    }
  }

  const auto new_train_catalog_fn = catalog_ver2_dir / "training_catalog_ver2.txt";
  std::ofstream f_train (new_train_catalog_fn);
  for (size_t i = 0UL; i < training_vec.size(); i++)
  {
    const auto & meta = training_vec[i];
    if (i != 0UL) { f_train << "\n"; }
    f_train << fmt::format("{} {:.8f} {:.8f}", meta.path, meta.northing, meta.easting);
  }
  f_train.close();
  spdlog::info("n_training: {}", training_vec.size());

  const auto new_query_catalog_fn = catalog_ver2_dir / "queries_catalog_ver2.txt";
  std::ofstream f_query (new_query_catalog_fn);
  for (size_t i = 0UL; i < queries_vec.size(); i++)
  {
    const auto & meta = queries_vec[i];
    if (i != 0UL) { f_query << "\n"; }
    f_query << fmt::format("{} {:.8f} {:.8f}", meta.path, meta.northing, meta.easting);
  }
  f_query.close();
  spdlog::info("n_queries: {}", queries_vec.size());

  for (int i = 0; i < 2; i++)
  {
    const auto new_database_catalog_fn = catalog_ver2_dir / fmt::format("database_catalog_ver2_aerial_{}", i);
    std::ofstream f_database (new_database_catalog_fn);
    for (size_t j = 0UL; j < aerial_database_vec[i].size(); j++)
    {
      const auto & meta = aerial_database_vec[i][j];
      if (j != 0UL) { f_database << "\n"; }
      f_database << fmt::format("{} {:.8f} {:.8f}", meta.path, meta.northing, meta.easting);
    }
    f_database.close();
    spdlog::info("n_database_a{}: {}", i, aerial_database_vec[i].size());
  }

  for (int i = 0; i < 19; i++)
  {
    const auto new_database_catalog_fn = catalog_ver2_dir / fmt::format("database_catalog_ver2_ground_{}", i);
    std::ofstream f_database (new_database_catalog_fn);
    for (size_t j = 0UL; j < ground_database_vec[i].size(); j++)
    {
      const auto & meta = ground_database_vec[i][j];
      if (j != 0UL) { f_database << "\n"; }
      f_database << fmt::format("{} {:.8f} {:.8f}", meta.path, meta.northing, meta.easting);
    }
    f_database.close();
    spdlog::info("n_database_g{}: {}", i, ground_database_vec[i].size());
  }

  /* Initialize Visualizer */
  // cwcloud::CloudVisualizer vis("explore_cs_campus", 1, 1);
  // const auto IDX_MIN = 0UL;
  // const auto IDX_MAX = meta_list.size()-1UL;
  // size_t idx = 0UL;

  // while (!vis.wasStopped())
  // {
  //   { /*  Register Query Cloud  & Clean other grids */
  //     const auto scan_fn = benchmark_dir / meta_list[idx].path;
  //     const auto scan = readCloudXYZ64(scan_fn.string());
  //     // spdlog::info("Scan path '{}' | n_points: {}", scan_fn.string(), scan.size());
  //     spdlog::info(
  //       "[{:04d}-Scan] north({:.4f}), east({:.4f})",
  //       idx,
  //       meta_list[idx].northing,
  //       meta_list[idx].easting);
  //     vis.setCloudXYZ(scan, 0, 0);
  //   }
  //   vis.run();
  //   if (vis.getKeySym() == "Right" && idx != IDX_MAX) { ++idx; }
  //   else if (vis.getKeySym() == "Left" && idx != IDX_MIN) { --idx; }
  // }
}


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
    const auto northing = std::stod(std::string(csv_row[1]));
    const auto easting  = std::stod(std::string(csv_row[2]));

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
