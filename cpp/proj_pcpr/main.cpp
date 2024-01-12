//
// Author: Chanwoo Lee (https://github.com/ChanWoo25)
//
#include <pcpr.hpp>
#include <cwcloud/CloudVisualizer.hpp>
#include <preprocess.hpp>

#include <argparse/argparse.hpp>
#include <spdlog/fmt/bundled/core.h>
#include <spdlog/fmt/bundled/format.h>
#include <spdlog/spdlog.h>

#include <filesystem>
namespace fs = std::filesystem;

using argparse::ArgumentParser;

void main_preprocess(const ArgumentParser & args)
{
  const auto dataset = args.get<std::string>("dataset");
  const auto root_dir = fs::path(args.get<std::string>("root_dir"));
  const auto seqs = args.get<std::vector<std::string>>("--seq");  // {"red", "green", "blue"}

  const auto mode = args.get<std::string>("mode");
  const auto n_samples = args.get<int>("n_samples");
  const auto ground = args.get<double>("ground");
  const auto sphere = args.get<double>("sphere");
  const auto interval = args.get<double>("interval");

  if (dataset == "boreas")
  {
    pcpr::preprocess_boreas(
      root_dir,
      seqs,
      mode,
      n_samples,
      ground,
      sphere,
      interval);
  }
}

void main_show(const ArgumentParser & args)
{
  const auto dataset = args.get<std::string>("dataset");
  const auto root_dir = fs::path(args.get<std::string>("root_dir"));
  const auto ground = args.get<double>("ground");
  const auto sphere = args.get<double>("sphere");
  const auto interval = args.get<double>("interval");
  cwcloud::CloudVisualizer vis("str", 1, 1);

  if (dataset == "boreas")
  {
    const auto list_fn = root_dir / "applanix" / "lidar_poses.csv";
    const auto lists = pcpr::read_scan_list_boreas(list_fn, interval);
    const int MIN_IDX = 0;
    const int MAX_IDX = static_cast<int>(lists.size());
    int scan_idx = 0;
    while(!vis.wasStopped())
    {
      const auto scan_path = root_dir / "lidar" / lists[scan_idx].path;
      const auto scan_fn = scan_path.string();
      auto cloud = pcpr::read_boreas_scan(scan_fn, ground, sphere);
      spdlog::info(
        "Read {} | n_pts: {}",
        scan_fn, cloud.size());

      vis.setCloudXYZ(cloud, 0, 0);
      vis.run();

      if (vis.getKeySym() == "Right")
      {
        scan_idx = std::min(scan_idx+1, MAX_IDX);
      }
      else if (vis.getKeySym() == "Left")
      {
        scan_idx = std::max(scan_idx-1, MIN_IDX);
      }
    }
  }
}

auto main(int argc, char * argv[]) -> int32_t
{
  ArgumentParser program("main");

  ArgumentParser command_preprocess("preprocess");
  command_preprocess.add_description(
    "Preprocess datasets to use in place recognition");
  command_preprocess.add_argument("dataset")
    .help("The name of dataset");
  command_preprocess.add_argument("root_dir")
    .help("The absolute(recommend) path of the root directory");
  command_preprocess.add_argument("-m", "--mode")
    .help("How to parse? [voxel/farthest]")
    .default_value("voxel");
  command_preprocess.add_argument("-n", "--n_samples")
    .help("The number of points to remain")
    .default_value(4096)
    .scan<'i', int>();
  command_preprocess.add_argument("-g", "--ground")
    .help("The height of ground in meter")
    .default_value(-3.0)
    .scan<'g', double>();
  command_preprocess.add_argument("-s", "--sphere")
    .help("The radius of accept sphere")
    .default_value(180.0)
    .scan<'g', double>();
  command_preprocess.add_argument("-i", "--interval")
    .help("The sampling interval length in meter")
    .default_value(-1.0)
    .scan<'g', double>();
  command_preprocess.add_argument("--seq")
    .default_value<std::vector<std::string>>({""})
    .append()
    .help("List sequense names you want to preprocess.");

  ArgumentParser command_show("show");
  command_show.add_description(
    "Show point clouds in a given directory");
  command_show.add_argument("dataset")
    .help("The name of dataset");
  command_show.add_argument("root_dir")
    .help("The absolute(recommend) path of the root directory");
  command_show.add_argument("-g", "--ground")
    .help("The height of ground in meter")
    .default_value(-2.0)
    .scan<'g', double>();
  command_show.add_argument("-s", "--sphere")
    .help("The radius of accept sphere")
    .default_value(180.0)
    .scan<'g', double>();
  command_show.add_argument("-i", "--interval")
    .help("The sampling interval length in meter")
    .default_value(10.0)
    .scan<'g', double>();


  program.add_subparser(command_preprocess);
  program.add_subparser(command_show);

  try
  {
    program.parse_args(argc, argv);
  }
  catch (const std::exception & err)
  {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    exit(1);
  }

  if (program.is_subcommand_used("preprocess"))
  {
    spdlog::info("[main-preprocess] Starts ...");
    const auto & args = program.at<ArgumentParser>("preprocess");
    main_preprocess(args);
  }
  else if (program.is_subcommand_used("show"))
  {
    spdlog::info("[main-show] Starts ...");
    const auto & args = program.at<ArgumentParser>("show");
    main_show(args);
  }
  else
  {
    std::cerr << program;
  }

  return EXIT_SUCCESS;
}
