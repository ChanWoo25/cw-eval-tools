// #include <opencv2/opencv.hpp>
// #include <cv_bridge/cv_bridge.h>

#include <memory>
#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/sample_consensus/method_types.h>
#include <spdlog/common.h>
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

std::vector<double> timestamps;
std::vector<Eigen::Matrix3d> rotations;
std::vector<Eigen::Quaterniond> quaternions;
std::vector<Eigen::Vector3d> translations;
std::vector<std::string> scan_fns;
size_t N_SCAN = 0UL;

void readPoses(const std::string & seq_dir)
{
  auto data_fn = fmt::format("{}/pose_fast_lio2.txt", seq_dir);
  std::fstream f_data(data_fn.c_str(), std::ios::in);
  // ROS_ASSERT(f_data.is_open());
  std::string line;
  while (std::getline(f_data, line))
  {
    if (line.empty()) { break; }
    std::istringstream iss(line);
    double timestamp;
    Eigen::Quaterniond quat;
    Eigen::Vector3d tran;
    iss >> timestamp;
    iss >> tran(0) >> tran(1) >> tran(2);
    iss >> quat.x() >> quat.y() >> quat.z() >> quat.w();
    timestamps.push_back(timestamp);
    quaternions.push_back(quat);
    rotations.push_back(quat.toRotationMatrix());
    translations.push_back(tran);
  }
  f_data.close();

  // ROS_ASSERT(timestamps.size() != 0UL);
  // ROS_ASSERT(timestamps.size() == translations.size());
  // ROS_ASSERT(timestamps.size() == quaternions.size());
  // ROS_ASSERT(timestamps.size() == rotations.size());
  N_SCAN = timestamps.size();
  fmt::print("[readPoses] N_SCAN (updated): {}\n", N_SCAN);
}

void readLidarData(const std::string & seq_dir)
{
  auto data_fn = fmt::format("{}/lidar_data.txt", seq_dir);
  std::fstream f_data(data_fn.c_str(), std::ios::in);
  std::string line;
  while (std::getline(f_data, line))
  {
    if (line.empty()) { break; }
    std::stringstream ss;
    ss << line;
    double timestamp;
    std::string scan_fn;

    ss >> timestamp >> scan_fn;
    scan_fns.push_back(fmt::format("{}/{}", seq_dir, scan_fn));
    if (scan_fns.size() == timestamps.size()) { break; }
  }
  f_data.close();

  // ROS_ASSERT(scan_fns.size() == timestamps.size());
  fmt::print("[readLidarData] 1st Scan '{}'\n", scan_fns.front());
  fmt::print("[readLidarData] Fin Scan '{}'\n", scan_fns.back());
}

#define HASH_P 116101
#define MAX_N 10000000000

class VOXEL_LOC {
public:
  int64_t x, y, z;

  VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
      : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOC &other) const {
    return (x == other.x && y == other.y && z == other.z);
  }
};

// for down sample function
struct M_POINT {
  float xyz[3];
  float intensity;
  int count = 0;
};

// Hash value

template <> struct std::hash<VOXEL_LOC> {
  int64_t operator()(const VOXEL_LOC &s) const {
    using std::hash;
    using std::size_t;
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
  }
};

auto voxelizedDownsampling(
  const pcl::PointCloud<pcl::PointXYZI> & cloud,
  const double & voxel_size)
  -> pcl::PointCloud<pcl::PointXYZI>
{
  /* Construct Voxel Map => Reduced Size */
  std::unordered_map<VOXEL_LOC, M_POINT> voxel_map;
  size_t plsize = cloud.size();
  for (size_t i = 0; i < plsize; i++)
  {
    const pcl::PointXYZI & p_c = cloud[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = p_c.data[j] / voxel_size;
      if (loc_xyz[j] < 0)
      {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position(
      (int64_t)loc_xyz[0],
      (int64_t)loc_xyz[1],
      (int64_t)loc_xyz[2]);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end())
    {
      iter->second.xyz[0] += p_c.x;
      iter->second.xyz[1] += p_c.y;
      iter->second.xyz[2] += p_c.z;
      iter->second.intensity += p_c.intensity;
      iter->second.count++;
    }
    else
    {
      M_POINT anp;
      anp.xyz[0] = p_c.x;
      anp.xyz[1] = p_c.y;
      anp.xyz[2] = p_c.z;
      anp.intensity = p_c.intensity;
      anp.count = 1;
      voxel_map[position] = anp;
    }
  }
  /* Copy to return cloud */
  auto new_sz = voxel_map.size();
  pcl::PointCloud<pcl::PointXYZI> downsampled;
  downsampled.resize(new_sz);
  size_t i = 0UL;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); ++iter)
  {
    downsampled[i].x = iter->second.xyz[0] / iter->second.count;
    downsampled[i].y = iter->second.xyz[1] / iter->second.count;
    downsampled[i].z = iter->second.xyz[2] / iter->second.count;
    downsampled[i].intensity = iter->second.intensity / iter->second.count;
    i++;
  }
  return downsampled;
}

auto downsamplingUntilTargetNumPoints(
  const pcl::PointCloud<pcl::PointXYZI> & origin_cloud,
  const size_t & target_size)
  -> pcl::PointCloud<pcl::PointXYZI>
{
  double voxel_size = 1.001;
  auto downsampled = voxelizedDownsampling(origin_cloud, voxel_size);
  while (downsampled.size() < target_size)
  {
    voxel_size -= 0.025;
    if (voxel_size <= 0.001) { break; }
    downsampled = voxelizedDownsampling(origin_cloud, voxel_size);
  }
  while (downsampled.size() > target_size)
  {
    voxel_size += 0.025;
    downsampled = voxelizedDownsampling(origin_cloud, voxel_size);
  }
  /* Add extra random points */
  if (downsampled.size() < target_size)
  {
    size_t n_add = target_size - downsampled.size();
    std::random_device rd;
    std::mt19937 mt(rd());
    int low {0}, high {static_cast<int>(downsampled.size())-1};
    // ROS_ASSERT (low <= high);
    // Define a range for random integers (e.g., between 1 and 100)
    std::uniform_int_distribution<int> dist(low, high);
    for (size_t i = 0; i < n_add; ++i)
    {
      int ri = dist(mt);
      downsampled.push_back(downsampled.points[ri]);
    }
  }
  // fmt::print(
  //   "[downsamplingUntilTargetNumPoints] points {} >> {}\n",
  //   origin_cloud.size(), downsampled.size());
  return downsampled;
}

pcl::PointXYZI vec2point(const Eigen::Vector3d &vec) {
  pcl::PointXYZI pi;
  pi.x = vec[0];
  pi.y = vec[1];
  pi.z = vec[2];
  return pi;
}

Eigen::Vector3d point2vec(const pcl::PointXYZI &pi) {
  return Eigen::Vector3d(pi.x, pi.y, pi.z);
}

auto readCloud(
  const std::string & scan_fn)
  ->pcl::PointCloud<pcl::PointXYZI>
{
  std::fstream f_bin(scan_fn, std::ios::in | std::ios::binary);
  // ROS_ASSERT(f_bin.is_open());

  f_bin.seekg(0, std::ios::end);
  const size_t num_elements = f_bin.tellg() / sizeof(float);
  std::vector<float> buf(num_elements);
  f_bin.seekg(0, std::ios::beg);
  f_bin.read(
    reinterpret_cast<char *>(
      &buf[0]),
      num_elements * sizeof(float));
  f_bin.close();
  // fmt::print("Add {} pts\n", num_elements/4);
  pcl::PointCloud<pcl::PointXYZI> cloud;
  for (std::size_t i = 0; i < buf.size(); i += 4)
  {
    pcl::PointXYZI point;
    point.x = buf[i];
    point.y = buf[i + 1];
    point.z = buf[i + 2];
    point.intensity = buf[i + 3];
    Eigen::Vector3d pv = point2vec(point);
    point = vec2point(pv);
    cloud.push_back(point);
  }
  return cloud;
}

auto getCloudMeanStd(
  const pcl::PointCloud<pcl::PointXYZI> & cloud)
  ->std::tuple<Eigen::Vector3f, float>
{
  if (cloud.points.empty()) { return {Eigen::Vector3f::Zero(), 0.0f}; }
  const auto n_pts = static_cast<float>(cloud.size());
  Eigen::Vector3f center = Eigen::Vector3f::Zero();
  for (const auto & point: cloud.points)
  {
    center(0) += point.x;
    center(1) += point.y;
    center(2) += point.z;
  }
  center /= n_pts;
  auto stddev = 0.0f;
  for (const auto & point: cloud.points)
  {
    stddev += std::sqrt(
      std::pow(point.x - center(0), 2.0f) +
      std::pow(point.y - center(1), 2.0f) +
      std::pow(point.z - center(2), 2.0f));
  }
  stddev /= n_pts;
  return {center, stddev};
}

auto normalizeCloud(
  const pcl::PointCloud<pcl::PointXYZI> & cloud)
  -> pcl::PointCloud<pcl::PointXYZI>
{
  const auto N = cloud.size();
  const auto [center, d] = getCloudMeanStd(cloud);
  pcl::PointCloud<pcl::PointXYZI> normalized;
  normalized.reserve(N);
  for (const auto & point: cloud.points)
  {
    auto nx = (point.x - center(0)) / (2.0f * d);
    auto ny = (point.y - center(1)) / (2.0f * d);
    auto nz = (point.z - center(2)) / (2.0f * d);
    if (   -1.0 <= nx && nx <= 1.0
        && -1.0 <= ny && ny <= 1.0
        && -1.0 <= nz && nz <= 1.0)
    {
      pcl::PointXYZI point;
      point.x = nx;
      point.y = ny;
      point.z = nz;
      normalized.push_back(point);
    }
  }

  /* Add extra random points */
  if (normalized.size() < N)
  {
    size_t n_add = N - normalized.size();
    // fmt::print("[normalizeCloud] Still {} points are required.\n", n_add);
    std::random_device rd;
    std::mt19937 mt(rd());
    int low {0}, high {static_cast<int>(normalized.size())-1};
    // ROS_ASSERT (low <= high);
    // Define a range for random integers (e.g., between 1 and 100)
    std::uniform_int_distribution<int> dist(low, high);
    for (size_t i = 0; i < n_add; ++i)
    {
      int ri = dist(mt);
      normalized.push_back(normalized.points[ri]);
    }
  }
  // ROS_ASSERT(normalized.size() == N);
  return normalized;
}

void writeScanBinary(
  const std::string & scan_fn,
  const pcl::PointCloud<pcl::PointXYZI> & cloud)
{
  std::ofstream f_scan(scan_fn, std::ios::out | std::ios::binary);
  // ROS_ASSERT(f_scan.is_open());
  for (const auto & point : cloud.points)
  {
    f_scan.write(reinterpret_cast<const char *>(&point.x), sizeof(float));
    f_scan.write(reinterpret_cast<const char *>(&point.y), sizeof(float));
    f_scan.write(reinterpret_cast<const char *>(&point.z), sizeof(float));
    f_scan.write(reinterpret_cast<const char *>(&point.intensity), sizeof(float));
  }
  f_scan.close();
}

void processSequence(
  const std::string & seq_dir,
  const size_t & target_num_points)
{
  N_SCAN = 0UL;
  timestamps.clear();
  translations.clear();
  quaternions.clear();
  rotations.clear();
  scan_fns.clear();
  readPoses(seq_dir);
  readLidarData(seq_dir);

  auto keyframe_data_fn = fmt::format("{}/lidar_keyframe_data.txt", seq_dir);
  auto keyframe_data_dir = fmt::format("{}/lidar_keyframe_downsampled", seq_dir);
  fs::create_directories(keyframe_data_dir);
  std::fstream f_key_data (keyframe_data_fn, std::ios::out);

  size_t prev_idx {10UL};
  size_t curr_idx {prev_idx};
  // Eigen::Vector3d prev_pos = Eigen::Vector3d::Zero();
  // Eigen::Vector3d curr_pos = Eigen::Vector3d::Zero();
  for (curr_idx = prev_idx; curr_idx < N_SCAN; ++curr_idx)
  {
    if (prev_idx == curr_idx)
    {
      auto & tran = translations[curr_idx];
      auto & quat = quaternions[curr_idx];
      auto line = fmt::format(
        "{:.6f} lidar_keyframe_downsampled/{:06d}.bin {:6f} {:6f} {:6f} {:6f} {:6f} {:6f} {:6f}\n",
        timestamps[curr_idx], curr_idx,
        tran(0), tran(1), tran(2),
        quat.x(), quat.y(), quat.z(), quat.w());
      f_key_data << line;
      /* Downsample cloud into target number of points */
      auto cloud = readCloud(scan_fns[curr_idx]);
      auto downsampled = downsamplingUntilTargetNumPoints(cloud, target_num_points);
      auto normalized = normalizeCloud(downsampled);
      auto new_scan_fn = fmt::format("{}/{:06d}.bin", keyframe_data_dir, curr_idx);
      writeScanBinary(new_scan_fn, normalized);
      fmt::print("[processSequence] Write to {}\n", new_scan_fn);
    }
    else
    {
      auto & prev_t = translations[prev_idx];
      auto & curr_t = translations[curr_idx];
      auto dist = (curr_t - prev_t).norm();
      auto & prev_R = rotations[prev_idx];
      auto & curr_R = rotations[curr_idx];
      Eigen::Matrix3d dist_R = prev_R.transpose() * curr_R;
      Eigen::Vector3d dist_r = Eigen::AngleAxisd(dist_R).angle() * Eigen::AngleAxisd(dist_R).axis();
      if (dist >= 0.5
          || dist_r(0) >= (M_PI * (10.0/180.0))
          || dist_r(1) >= (M_PI * (10.0/180.0))
          || dist_r(2) >= (M_PI * (10.0/180.0)))
      {
        // fmt::print(
        //   "dist({:.3f}), droll({:.3f}), dpitch({:.3f}), dyaw({:.3f})\n",
        //   dist, dist_r(0), dist_r(1), dist_r(2));

        auto & tran = translations[curr_idx];
        auto & quat = quaternions[curr_idx];
        auto line = fmt::format(
          "{:.6f} lidar_keyframe_downsampled/{:06d}.bin {:6f} {:6f} {:6f} {:6f} {:6f} {:6f} {:6f}\n",
          timestamps[curr_idx], curr_idx,
          tran(0), tran(1), tran(2),
          quat.x(), quat.y(), quat.z(), quat.w());
        f_key_data << line;
        /* Downsample cloud into target number of points */
        auto cloud = readCloud(scan_fns[curr_idx]);
        auto downsampled = downsamplingUntilTargetNumPoints(cloud, target_num_points);
        auto normalized = normalizeCloud(downsampled);
        auto new_scan_fn = fmt::format("{}/{:06d}.bin", keyframe_data_dir, curr_idx);
        writeScanBinary(new_scan_fn, normalized);
        fmt::print("[processSequence] Write to {}\n", new_scan_fn);
        prev_idx = curr_idx;
      }
    }
  }
  f_key_data.close();
}


cwcloud::CloudVisualizer vis("str", 1, 2);

void processHaomoSequence(
  const std::string & seq_dir,
  const size_t & target_num_points)
{
  fs::path root_dir(seq_dir);
  // fs::create_directory(root_dir.parent_path()/"")
  // spdlog::info("parent_path: {}", root_dir.parent_path().string());
  // spdlog::info("filename: {}", root_dir.filename().string());
  auto save_dir = root_dir.parent_path() / (root_dir.filename().string() + "_downsampled");
  spdlog::info("save_dir: {}", save_dir.string());
  fs::create_directory(save_dir);
  // spdlog::info("current_path: {}", fs::current_path().string());
  // spdlog::info("relative_path: {}", root_dir.relative_path().string());
  // spdlog::info("absolute_path: {}", fs::absolute(root_dir).string());
  // spdlog::info("canonical_path: {}", fs::canonical(root_dir).string());

  if (fs::exists(root_dir) && fs::is_directory(root_dir))
  {
    auto dit = fs::directory_iterator(root_dir);
    for (const auto & entry : dit)
    {
      const auto scan_fn = entry.path().string();
      const auto new_scan_fn = (save_dir / entry.path().filename()).string();
      // spdlog::info(scan_fn);
      spdlog::info("Process {} ...", new_scan_fn);

      auto cloud = readCloud(scan_fn);

      /* Segment Ground */
      auto coefficients = std::make_shared<pcl::ModelCoefficients>();
      auto inliers = std::make_shared<pcl::PointIndices>();
      // pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
      // pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
      pcl::SACSegmentation<pcl::PointXYZI> seg;
      seg.setOptimizeCoefficients(true);
      seg.setModelType(pcl::SACMODEL_PLANE);
      seg.setMethodType(pcl::SAC_RANSAC);
      // seg.setMaxIterations(1000);
      seg.setDistanceThreshold(0.5);
      seg.setInputCloud(cloud.makeShared());

      seg.segment(*inliers, *coefficients);

      pcl::ExtractIndices<pcl::PointXYZI> extract_ground;
      // pcl::ExtractIndices<pcl::PointXYZI> extract_inform;
      extract_ground.setInputCloud(cloud.makeShared());
      // extract_inform.setInputCloud(cloud.makeShared());
      extract_ground.setIndices(inliers);
      // extract_inform.setIndices(inliers);
      extract_ground.setNegative(false); // Extract the ground points
      // extract_inform.setNegative(true); // Extract the ground points
      auto ground_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
      auto inform_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
      extract_ground.filter(*ground_cloud);

      extract_ground.setNegative(true); // Extract the ground points
      extract_ground.filter(*inform_cloud);
      spdlog::info("ground points: {}", ground_cloud->points.size());
      spdlog::info("not ground points: {}", inform_cloud->points.size());
      // extract_inform.filter(*inform_cloud);

      auto downsampled = downsamplingUntilTargetNumPoints(*inform_cloud, target_num_points);
      auto normalized = normalizeCloud(downsampled);
      writeScanBinary(new_scan_fn, normalized);

      vis.setGroundInformCloud(*ground_cloud, *inform_cloud, "origin", 0);
      vis.setCloud(downsampled, "downsampled", 1); // , entry.path().filename().string()
      vis.run();
    }
  }

  return;
  // N_SCAN = 0UL;
  // timestamps.clear();
  // translations.clear();
  // quaternions.clear();
  // rotations.clear();
  // scan_fns.clear();
  // readPoses(seq_dir);
  // readLidarData(seq_dir);

  // auto keyframe_data_fn = fmt::format("{}/lidar_keyframe_data.txt", seq_dir);
  // auto keyframe_data_dir = fmt::format("{}/lidar_keyframe_downsampled", seq_dir);
  // fs::create_directories(keyframe_data_dir);
  // std::fstream f_key_data (keyframe_data_fn, std::ios::out);

  // size_t prev_idx {10UL};
  // size_t curr_idx {prev_idx};
  // // Eigen::Vector3d prev_pos = Eigen::Vector3d::Zero();
  // // Eigen::Vector3d curr_pos = Eigen::Vector3d::Zero();
  // for (curr_idx = prev_idx; curr_idx < N_SCAN; ++curr_idx)
  // {
  //   if (prev_idx == curr_idx)
  //   {
  //     auto & tran = translations[curr_idx];
  //     auto & quat = quaternions[curr_idx];
  //     auto line = fmt::format(
  //       "{:.6f} lidar_keyframe_downsampled/{:06d}.bin {:6f} {:6f} {:6f} {:6f} {:6f} {:6f} {:6f}\n",
  //       timestamps[curr_idx], curr_idx,
  //       tran(0), tran(1), tran(2),
  //       quat.x(), quat.y(), quat.z(), quat.w());
  //     f_key_data << line;
  //     /* Downsample cloud into target number of points */
  //     auto cloud = readCloud(scan_fns[curr_idx]);
  //     auto downsampled = downsamplingUntilTargetNumPoints(cloud, target_num_points);
  //     auto normalized = normalizeCloud(downsampled);
  //     auto new_scan_fn = fmt::format("{}/{:06d}.bin", keyframe_data_dir, curr_idx);
  //     writeScanBinary(new_scan_fn, normalized);
  //     fmt::print("[processSequence] Write to {}\n", new_scan_fn);
  //   }
  //   else
  //   {
  //     auto & prev_t = translations[prev_idx];
  //     auto & curr_t = translations[curr_idx];
  //     auto dist = (curr_t - prev_t).norm();
  //     auto & prev_R = rotations[prev_idx];
  //     auto & curr_R = rotations[curr_idx];
  //     Eigen::Matrix3d dist_R = prev_R.transpose() * curr_R;
  //     Eigen::Vector3d dist_r = Eigen::AngleAxisd(dist_R).angle() * Eigen::AngleAxisd(dist_R).axis();
  //     if (dist >= 0.5
  //         || dist_r(0) >= (M_PI * (10.0/180.0))
  //         || dist_r(1) >= (M_PI * (10.0/180.0))
  //         || dist_r(2) >= (M_PI * (10.0/180.0)))
  //     {
  //       // fmt::print(
  //       //   "dist({:.3f}), droll({:.3f}), dpitch({:.3f}), dyaw({:.3f})\n",
  //       //   dist, dist_r(0), dist_r(1), dist_r(2));

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
  //       prev_idx = curr_idx;
  //     }
  //   }
  // }
  // f_key_data.close();
}
struct Config
{
  int target_n_points = 4096;
} cfg ;

auto main() -> int32_t
{
  // cwcloud::CloudVisualizer vis("str");
  // vis.run();

  std::vector<std::string> sequences;
  // sequences.emplace_back("/data/datasets/dataset_project/hangpa00");
  // sequences.emplace_back("/data/datasets/dataset_project/hangpa01");
  // sequences.emplace_back("/data/datasets/dataset_project/hangpa02");
  // sequences.emplace_back("/data/datasets/dataset_project/outdoor00");
  // sequences.emplace_back("/data/datasets/dataset_project/outdoor01");
  // sequences.emplace_back("/data/datasets/dataset_project/outdoor02");
  // sequences.emplace_back("/data/datasets/dataset_project/outdoor03");
  // sequences.emplace_back("/data/datasets/dataset_project/itbt00");
  // sequences.emplace_back("/data/datasets/dataset_project/itbt01");
  // sequences.emplace_back("/data/datasets/dataset_project/itbt02");
  // sequences.emplace_back("/data/datasets/dataset_project/itbt03");
  // sequences.emplace_back("/data/datasets/dataset_project/itbt_dark00");
  // sequences.emplace_back("/data/datasets/dataset_project/itbt_dark01");
  // sequences.emplace_back("/data/datasets/dataset_project/itbt_dark02");
  // sequences.emplace_back("/data/datasets/dataset_project/itbt_dark03");
  // sequences.emplace_back("/data/datasets/dataset_haomo/01_02");
  sequences.emplace_back("/data/datasets/dataset_haomo/03");

  for (const auto & seq_dir: sequences)
  {
    fmt::print("Process {} ...\n", seq_dir);
    // processSequence(seq_dir, cfg.target_n_points);
    processHaomoSequence(seq_dir, cfg.target_n_points);
  }

  return EXIT_SUCCESS;
};
