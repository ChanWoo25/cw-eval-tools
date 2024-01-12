#ifndef PREPROCESS_HPP_
#define PREPROCESS_HPP_

#include <spdlog/fmt/bundled/core.h>
#include <spdlog/fmt/bundled/format.h>
#include <spdlog/spdlog.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

/* For plane segmentation */
#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <string>
#include <memory>
#include <fstream>
#include <vector>
#include <algorithm>
#include <tuple>
#include <filesystem>
#include <random>

namespace fs = std::filesystem;

namespace pcpr {

auto read_boreas_scan(
  const std::string & scan_fn)
  ->pcl::PointCloud<pcl::PointXYZI>;

void preprocess_boreas(
  const fs::path & root_dir,
  const std::string & mode,
  const int & n_samples,
  const double & interval);

}; // namespace pcpr

#endif
