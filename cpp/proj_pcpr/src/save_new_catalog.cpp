
#include <cwcloud/CloudVisualizer.hpp>
#include <pcpr.hpp>

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
namespace fs = std::filesystem;

fs::path datasets_dir("/data/datasets");
const auto cs_campus_dir = datasets_dir / "dataset_cs_campus";
const auto benchmark_dir = cs_campus_dir / "benchmark_datasets";
const auto umd_dir = benchmark_dir / "umd";
const auto debug_dir = cs_campus_dir / "debug";

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
  const std::vector<MetaPerScan> & dbase_metas,
  const int k)
  -> std::tuple<std::vector<size_t>, std::vector<double>>
{
  if (k < 1) { spdlog::error("k({}) < 1 not allowed.", k); exit(1); }

  std::vector<size_t> top_k_indices(k, 0UL);
  std::vector<double> top_k_dists  (k, std::numeric_limits<double>::max());
  for (size_t i = 0UL; i < dbase_metas.size(); i += 2UL)
  {
    const auto dist = meta_dist(ground, dbase_metas[i]);

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
  const std::vector<MetaPerScan> & query_metas,
  const std::vector<MetaPerScan> & dbase_metas,
  const int k)
{
  spdlog::info ("save_fn: {}, top_k: {}", save_fn.string(), k);
  std::ofstream f_out(save_fn);
  for (size_t i = 0; i < query_metas.size(); ++i)
  {
    const auto & meta = query_metas[i];
    const auto [top_k_indices, top_k_dists]
      = find_top_k_neighbors(meta, dbase_metas, k);

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
  // add_command.add_argument("--dbase")
  //   .help("If dbase is give, get nearest against ground dataset")
  //   .default_value(-1)
  //   .scan<'i', int>();

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
    = cs_campus_dir / "catalog-nearest";
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

  spdlog::info("[readCatalog]");
  const auto [dbase_catalog, query_catalog]
    = readCatalog(cs_campus_dir / "catalog");
  for (uint32_t di = 0U; di < dbase_catalog.size(); di++)
  {
    spdlog::info("- {}-th db, length: {}", di, dbase_catalog[di].size());
  }
  spdlog::info("- qeuries, length: {}", query_catalog.size());

  for (size_t i = 0UL; i < dbase_catalog.size(); ++i) // queries
  {
    constexpr int top_k = 4;
    const auto save_fn
      = catalog_dir
        / fmt::format(
            "query-dbase{:02d}-top_{}-nearest.txt",
            i, top_k);
    find_n_save_top_k_neighbors(
      save_fn,
      query_catalog,
      dbase_catalog[i],
      top_k);
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
  //     const auto scan = readCloudXyz64(scan_fn.string());
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
