#include <pcpr.hpp>
#include <cwcloud/CloudVisualizer.hpp>

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
#include <argparse/argparse.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <tuple>
#include <random>
#include <filesystem>
namespace fs = std::filesystem;


const auto datasets_dir  = fs::path("/data/datasets");
const auto cs_campus_dir = datasets_dir / "dataset_cs_campus";
const auto benchmark_dir = cs_campus_dir / "benchmark_datasets";
const auto debug_dir     = cs_campus_dir / "debug";
const auto catalog_dir   = cs_campus_dir / "catalog";
const auto umd_dir       = benchmark_dir / "umd";

void main_with_nearest(
  const int & main_seq,
  const int & sub_seq,
  const double & skip_len,
  const double & scale=1.0);

void main_scaling(
  const double & scale=1.0);

auto main(int argc, char * argv[]) -> int32_t
{
  argparse::ArgumentParser program("explore_cs_campus");

  /// explore_cs_campus with_nearest dbase_index
  argparse::ArgumentParser with_nearest_command("with_nearest");
  with_nearest_command.add_description(
    "show scans with nearest other sequence\n"
    "Aerial: 0, 1\n"
    "Ground: 2 ~ 20");
  with_nearest_command.add_argument("main_seq")
    .help("Main sequence to visualize")
    .scan<'i', int>();
  with_nearest_command.add_argument("--sub_seq")
    .help("Sub sequence to visualize. If no sub_seq is given, \n"
          "find nearest aerial scan from 0 & 1")
    .default_value(-1)
    .scan<'i', int>();
  with_nearest_command.add_argument("--scale")
    .help("Sub sequence to visualize")
    .default_value(1.0)
    .scan<'g', double>();
  with_nearest_command.add_argument("--skip_len")
    .help("Skip Length")
    .default_value(-1.0)
    .scan<'g', double>();

  argparse::ArgumentParser scaling_command("scaling");
  scaling_command.add_description(
    "show scans with nearest other sequence\n"
    "Aerial: 0, 1\n"
    "Ground: 2 ~ 20");
  scaling_command.add_argument("new_save_prefix")
    .help("prefix attched after created directory")
    .default_value("_new");
  scaling_command.add_argument("--scale")
    .help("Sub sequence to visualize")
    .default_value(1.0)
    .scan<'g', double>();

  program.add_subparser(with_nearest_command);
  program.add_subparser(scaling_command);

  try {
    program.parse_args(argc, argv);
  }
  catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return EXIT_FAILURE;
  }

  if (program.is_subcommand_used("with_nearest"))
  {
    const auto main_seq = with_nearest_command.get<int>("main_seq");
    const auto sub_seq = with_nearest_command.get<int>("sub_seq");
    const auto skip_len = with_nearest_command.get<double>("skip_len");
    const auto scale = with_nearest_command.get<double>("scale");
    if (!(0 <= main_seq && main_seq <= 20)) {
      spdlog::error("out of seq range | main_seq: {}", main_seq);
    }
    if (!(0 <= sub_seq && sub_seq <= 20)) {
      spdlog::error("out of seq range | sub_seq: {}", sub_seq);
    }
    if (!(scale <= 1.0)) {
      spdlog::error("'scale'({}) should be less than 1.0", scale);
    }
    main_with_nearest(
      main_seq,
      sub_seq,
      skip_len,
      scale);
  }
  else
  {
    const auto scale = scaling_command.get<double>("scale");
    if (!(scale <= 1.0)) {
      spdlog::error("'scale'({}) should be less than 1.0", scale);
    }
    main_scaling(scale);
  }

  return EXIT_SUCCESS;
}

void main_with_nearest(
  const int & main_seq,
  const int & sub_seq,
  const double & skip_len,
  const double & scale)
{
  const auto main_dir
    = (main_seq == 0 || main_seq == 1) // is aerial
    ? (umd_dir / fmt::format("umcp_lidar5_cloud_{}", (main_seq+6)))
    : (umd_dir / fmt::format("umcp_lidar5_ground_umd_gps_{}", (main_seq-1)));
  const auto main_list_fn = main_dir / "umd_aerial_cloud_20m_100coverage_4096.csv";

  spdlog::info("Read {}.", main_list_fn.string());
  const auto meta_list = readScanList(main_list_fn, skip_len);

  /* Initialize Visualizer */
  const auto IDX_MIN = 0UL;
  const auto IDX_MAX = meta_list.size()-1UL;
  size_t idx = 0UL;

  if (sub_seq < 0)
  {
    if (main_seq < 2 || 20 < main_seq) {
      spdlog::error("main_seq({}) must be in [2, 20] when sub_seq is not given.", main_seq);
    }

    cwcloud::CloudVisualizer vis("explore_cs_campus", 1, 2);
    const auto aerial00_fn = umd_dir / "umcp_lidar5_cloud_6" / "umd_aerial_cloud_20m_100coverage_4096.csv";
    const auto aerial01_fn = umd_dir / "umcp_lidar5_cloud_7" / "umd_aerial_cloud_20m_100coverage_4096.csv";
    const auto nearest00_fn = cs_campus_dir / "catalog-nearest" / fmt::format("ground{:02d}-aerial00-top_4-nearest.txt", (main_seq-2));
    const auto nearest01_fn = cs_campus_dir / "catalog-nearest" / fmt::format("ground{:02d}-aerial01-top_4-nearest.txt", (main_seq-2));
    spdlog::info("Read \"{}\" | {}", aerial00_fn.string(), fs::exists(aerial00_fn));
    spdlog::info("Read \"{}\" | {}", aerial01_fn.string(), fs::exists(aerial01_fn));
    const auto aerial00_list = readScanList(aerial00_fn);
    const auto aerial01_list = readScanList(aerial01_fn);
    spdlog::info("Read \"{}\" | {}", nearest00_fn.string(), fs::exists(nearest00_fn));
    spdlog::info("Read \"{}\" | {}", nearest01_fn.string(), fs::exists(nearest01_fn));
    const auto nearest00_list = readNearestMetas(nearest00_fn);
    const auto nearest01_list = readNearestMetas(nearest01_fn);

    while (!vis.wasStopped())
    {
      const auto scan_fn = benchmark_dir / meta_list[idx].path;
      const auto scan = readCloudXyz64(scan_fn.string());
      spdlog::info(
        "[{:04d}-Scan] '{}' | north({:.4f}), east({:.4f})",
        idx,
        scan_fn.string(),
        meta_list[idx].northing,
        meta_list[idx].easting);
      vis.setCloudXYZ(scan, 0, 0);

      /* choose nearest */
      const auto & near00 = nearest00_list[idx];
      const auto & near01 = nearest01_list[idx];
      int aseq = 0;
      size_t nn_idx;
      double nn_dist = std::numeric_limits<double>::max();
      for (size_t i = 0UL; i < near00.size(); ++i)
      {
        if (near00.top_k_dists[i] < nn_dist)
        {
          aseq = 0;
          nn_idx = near00.top_k_indices[i];
          nn_dist = near00.top_k_dists[i];
        }
      }
      for (size_t i = 0UL; i < near01.size(); ++i)
      {
        if (near01.top_k_dists[i] < nn_dist)
        {
          aseq = 1;
          nn_idx = near01.top_k_indices[i];
          nn_dist = near01.top_k_dists[i];
        }
      }
      const auto nn_scan_fn
        = (aseq == 0)
        ? (benchmark_dir / aerial00_list[nn_idx].path)
        : (benchmark_dir / aerial01_list[nn_idx].path);
      auto nn_scan = readCloudXyz64(nn_scan_fn.string());

      if (scale < 1.0)
      {
        pcl::PointCloud<pcl::PointXYZ> tmp_scan;
        const auto inv_scale = 1.0 / scale;
        /* Re-downsampling */
        for (const auto & pt: nn_scan.points)
        {
          auto norm = std::sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
          if (norm < scale) {
            tmp_scan.emplace_back(pt.x * inv_scale,
                                  pt.y * inv_scale,
                                  pt.z * inv_scale);
          }
        }
        if (tmp_scan.size() <= 10) {
          spdlog::error("Something Wrong with scaling: reduced size: {}", tmp_scan.size());
          exit(1);
        }
        else {
          spdlog::info("Reduced to # {}", tmp_scan.size());
        }
        nn_scan = tmp_scan;
      }
      spdlog::info(
        "[Nearest] {} / dist: {:.6f}", nn_scan_fn.string(), nn_dist);
      vis.setCloudXYZwithNum(nn_scan, 0, 1, nn_dist, 2.0);
      vis.run();

      if      (vis.getKeySym() == "Right" && idx != IDX_MAX) { ++idx; }
      else if (vis.getKeySym() == "Left"  && idx != IDX_MIN) { --idx; }
    }
  }
  else
  {
    cwcloud::CloudVisualizer vis("explore_cs_campus", 1, 1);
    while (!vis.wasStopped())
    {
      { /*  Register Query Cloud  & Clean other grids */
        const auto scan_fn = benchmark_dir / meta_list[idx].path;
        const auto scan = readCloudXyz64(scan_fn.string());
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
  }
}

void main_scaling(
  const double & scale)
{
  spdlog::info("[main_scaling] : scale({:.2f})", scale);
  const auto old_dir_00 = umd_dir / "umcp_lidar5_cloud_6";
  const auto old_dir_01 = umd_dir / "umcp_lidar5_cloud_7";
  const std::string new_dir_name_00 = fmt::format(
    "umcp_aerial_resized{:03d}_00",
    static_cast<int>(100.0 * scale));
  const std::string new_dir_name_01 = fmt::format(
    "umcp_aerial_resized{:03d}_01",
    static_cast<int>(100.0 * scale));
  const auto new_dir_00 = umd_dir / new_dir_name_00;
  const auto new_dir_01 = umd_dir / new_dir_name_01;
  fs::create_directories(new_dir_00);
  fs::create_directories(new_dir_01);
  fs::create_directories(new_dir_00 / "bins_20m_100coverage_4096");
  fs::create_directories(new_dir_01 / "bins_20m_100coverage_4096");

  const auto old_list_00_fn = old_dir_00 / "umd_aerial_cloud_20m_100coverage_4096_new.csv";
  const auto old_list_01_fn = old_dir_01 / "umd_aerial_cloud_20m_100coverage_4096_new.csv";
  const auto new_list_00_fn = new_dir_00 / "umd_aerial_cloud_20m_100coverage_4096_new.csv";
  const auto new_list_01_fn = new_dir_01 / "umd_aerial_cloud_20m_100coverage_4096_new.csv";
  std::ofstream f_new_list_00(new_list_00_fn);
  std::ofstream f_new_list_01(new_list_01_fn);
  f_new_list_00 << "file,northing,easting";
  f_new_list_01 << "file,northing,easting";

  spdlog::info("old_list_00: {}.", old_list_00_fn.string());
  spdlog::info("old_list_01: {}.", old_list_01_fn.string());
  spdlog::info("new_list_00: {}.", new_list_00_fn.string());
  spdlog::info("new_list_01: {}.", new_list_01_fn.string());

  const auto old_list_00 = readScanList(old_list_00_fn);
  const auto old_list_01 = readScanList(old_list_01_fn);
  const auto new_list_00 = readScanList(new_list_00_fn);
  const auto new_list_01 = readScanList(new_list_01_fn);

  size_t scan_idx = 0UL;
  for (const auto & meta: old_list_00)
  {
    const std::string old_scan_name = meta.path.substr(50, 14);
    // const auto old_scan_idx  = std::stoi(meta.path.substr(50, 8));
    const auto old_scan_mod  = std::stoi(meta.path.substr(59, 1));
    // spdlog::info("old_scan_fn: {} / {} / {}",
    //   old_scan_name, old_scan_idx, old_scan_mod);
    const auto old_scan_fn = benchmark_dir / meta.path;
    const auto new_scan_fn
      = (old_scan_mod == 0)
      ? (new_dir_00 / "bins_20m_100coverage_4096" / fmt::format("{:08d}.bin", scan_idx))
      : (new_dir_01 / "bins_20m_100coverage_4096" / fmt::format("{:08d}.bin", scan_idx));
    const auto new_scan_line
      = (old_scan_mod == 0)
      ? (fmt::format("\numd/{}/bins_20m_100coverage_4096/{:08d}.bin,{:.9f},{:.9f}",
          new_dir_name_00,
          scan_idx,
          meta.northing,
          meta.easting))
      : (fmt::format("\numd/{}/bins_20m_100coverage_4096/{:08d}.bin,{:.9f},{:.9f}",
          new_dir_name_01,
          scan_idx,
          meta.northing,
          meta.easting));

    const auto old_scan = readCloudXyz64(old_scan_fn.string());
    // spdlog::info("new_scan_fn: {}, size: {}", new_scan_fn.string(), old_scan.size());
    // spdlog::info("{}", new_scan_line);
    const auto new_scan = scalingCloud(old_scan, scale);

    if (old_scan_mod == 0) {
      f_new_list_00 << new_scan_line;
    } else {
      f_new_list_01 << new_scan_line;
    }
    writeCloudXyz64(new_scan_fn.string(), new_scan);
    if (old_scan_mod == 1)
    {
      ++scan_idx;
    }
  }
  for (const auto & meta: old_list_01)
  {
    const std::string old_scan_name = meta.path.substr(50, 14);
    // const auto old_scan_idx  = std::stoi(meta.path.substr(50, 8));
    const auto old_scan_mod  = std::stoi(meta.path.substr(59, 1));
    // spdlog::info("old_scan_fn: {} / {} / {}",
    //   old_scan_name, old_scan_idx, old_scan_mod);
    const auto old_scan_fn = benchmark_dir / meta.path;
    const auto new_scan_fn
      = (old_scan_mod == 0)
      ? (new_dir_00 / "bins_20m_100coverage_4096" / fmt::format("{:08d}.bin", scan_idx))
      : (new_dir_01 / "bins_20m_100coverage_4096" / fmt::format("{:08d}.bin", scan_idx));
    const auto new_scan_line
      = (old_scan_mod == 0)
      ? (fmt::format("\numd/{}/bins_20m_100coverage_4096/{:08d}.bin,{:.9f},{:.9f}",
          new_dir_name_00,
          scan_idx,
          meta.northing,
          meta.easting))
      : (fmt::format("\numd/{}/bins_20m_100coverage_4096/{:08d}.bin,{:.9f},{:.9f}",
          new_dir_name_01,
          scan_idx,
          meta.northing,
          meta.easting));

    const auto old_scan = readCloudXyz64(old_scan_fn.string());
    // spdlog::info("{}", new_scan_line);
    const auto new_scan = scalingCloud(old_scan, scale);
    spdlog::info("new_scan_fn: {} ({})", new_scan_fn.string(), new_scan.size());

    if (old_scan_mod == 0) {
      f_new_list_00 << new_scan_line;
    } else {
      f_new_list_01 << new_scan_line;
    }
    writeCloudXyz64(new_scan_fn.string(), new_scan);
    if (old_scan_mod == 1)
    {
      ++scan_idx;
    }
  }

  f_new_list_00.close();
  f_new_list_01.close();
}
