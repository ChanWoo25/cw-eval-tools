#include <pcpr.hpp>
#include <random>
#include <spdlog/spdlog.h>

namespace pcpr {

auto vec3dToPtXyz(
  const Eigen::Vector3d & vec)
  -> pcl::PointXYZ
{
  pcl::PointXYZ pt;
  pt.x = vec[0];
  pt.y = vec[1];
  pt.z = vec[2];
  return pt;
}

auto vec3dToPtXyzi(
  const Eigen::Vector3d & vec)
  -> pcl::PointXYZI
{
  pcl::PointXYZI pi;
  pi.x = vec[0];
  pi.y = vec[1];
  pi.z = vec[2];
  return pi;
}

auto ptXyziToVec3d(
  const pcl::PointXYZI &pi)
  -> Eigen::Vector3d
{
  return Eigen::Vector3d(pi.x, pi.y, pi.z);
}

auto ptXyzToVec3d(
  const pcl::PointXYZ &pi)
  -> Eigen::Vector3d
{
  return Eigen::Vector3d(pi.x, pi.y, pi.z);
}

auto get_cloud_mean_std_xyz(
  const pcl::PointCloud<pcl::PointXYZ> & cloud)
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

auto normalize_cloud_xyz(
  const pcl::PointCloud<pcl::PointXYZ> & cloud)
  -> pcl::PointCloud<pcl::PointXYZ>
{
  const auto N = cloud.size();
  const auto [center, d] = get_cloud_mean_std_xyz(cloud);

  // spdlog::info("[normalize_cloud_xyz] mean: {}, std: {}", center, d);

  pcl::PointCloud<pcl::PointXYZ> normalized;
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
      pcl::PointXYZ point;
      point.x = nx;
      point.y = ny;
      point.z = nz;
      normalized.push_back(point);
    }
  }
  // spdlog::info("[normalize_cloud_xyz] Step 1 size: ", normalized.size());

  /* Add extra random points */
  if (normalized.size() < N)
  {
    size_t n_add = N - normalized.size();
    // fmt::print("[normalizeCloud] Still {} points are required.\n", n_add);
    std::random_device rd;
    std::mt19937 mt(rd());
    int low {0}, high {static_cast<int>(normalized.size())-1};

    // Define a range for random integers (e.g., between 1 and 100)
    std::uniform_int_distribution<int> dist(low, high);
    for (size_t i = 0; i < n_add; ++i)
    {
      int ri = dist(mt);
      normalized.push_back(normalized.points[ri]);
    }
  }
  // spdlog::info("[normalize_cloud_xyz] Step 2 size: ", normalized.size());

  return normalized;
}

auto voxelized_downsampling_xyz(
  const pcl::PointCloud<pcl::PointXYZ> & cloud,
  const double & voxel_size)
  -> pcl::PointCloud<pcl::PointXYZ>
{
  /* Construct Voxel Map => Reduced Size */
  std::unordered_map<VOXEL_LOC, M_POINT> voxel_map;
  size_t plsize = cloud.size();
  for (size_t i = 0; i < plsize; i++)
  {
    const pcl::PointXYZ & p_c = cloud[i];
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
      iter->second.count++;
    }
    else
    {
      M_POINT anp;
      anp.xyz[0] = p_c.x;
      anp.xyz[1] = p_c.y;
      anp.xyz[2] = p_c.z;
      anp.count = 1;
      voxel_map[position] = anp;
    }
  }
  /* Copy to return cloud */
  auto new_sz = voxel_map.size();
  pcl::PointCloud<pcl::PointXYZ> downsampled;
  downsampled.resize(new_sz);
  size_t i = 0UL;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); ++iter)
  {
    downsampled[i].x = iter->second.xyz[0] / iter->second.count;
    downsampled[i].y = iter->second.xyz[1] / iter->second.count;
    downsampled[i].z = iter->second.xyz[2] / iter->second.count;
    i++;
  }
  return downsampled;
}

void write_scan_bin_xyz(
  const std::string & scan_fn,
  const pcl::PointCloud<pcl::PointXYZ> & cloud)
{
  std::ofstream f_scan(scan_fn, std::ios::out | std::ios::binary);
  if (!f_scan.is_open()) {
    spdlog::error("{} is not opened!", scan_fn);
  }

  for (const auto & point : cloud.points)
  {
    f_scan.write(reinterpret_cast<const char *>(&point.x), sizeof(float));
    f_scan.write(reinterpret_cast<const char *>(&point.y), sizeof(float));
    f_scan.write(reinterpret_cast<const char *>(&point.z), sizeof(float));
  }
  f_scan.close();
}

void write_scan_bin_xyzi(
  const std::string & scan_fn,
  const pcl::PointCloud<pcl::PointXYZI> & cloud)
{
  std::ofstream f_scan(scan_fn, std::ios::out | std::ios::binary);
  if (!f_scan.is_open()) {
    spdlog::error("{} is not opened!", scan_fn);
  }
  for (const auto & point : cloud.points)
  {
    f_scan.write(reinterpret_cast<const char *>(&point.x), sizeof(float));
    f_scan.write(reinterpret_cast<const char *>(&point.y), sizeof(float));
    f_scan.write(reinterpret_cast<const char *>(&point.z), sizeof(float));
    f_scan.write(reinterpret_cast<const char *>(&point.intensity), sizeof(float));
  }
  f_scan.close();
}

auto strip(
  const std::string & str)
  -> std::string
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


auto readNearestMetas(
  const fs::path & data_fn)
  ->std::vector<NearestMeta>
{
  std::vector<NearestMeta> nearest_metas;
  std::ifstream fin(data_fn);
  std::string line;
  while (std::getline(fin, line))
  {
    line = strip(line);
    if (line.empty()) { break; }

    // static int i = 0;
    // if (i++ % 1000 == 0) { spdlog::info("line size: {}", line.size()); }
    std::istringstream iss(line);

    std::string path;
    double northing;
    double easting;
    int k;
    iss >> path;
    iss >> northing;
    iss >> easting;
    iss >> k;

    NearestMeta nearest_meta;
    size_t idx;
    double dist;
    for (int i = 0; i < k; ++i)
    {
      iss >> idx;
      iss >> dist;
      nearest_meta.top_k_indices.push_back(idx);
      nearest_meta.top_k_dists.push_back(dist);
    }
    nearest_metas.push_back(nearest_meta);
  }
  fin.close();

  spdlog::info("read size: {}", nearest_metas.size());
  return nearest_metas;
}

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

auto readScanList(
  const fs::path & list_fn,
  const double & skip_len)
  -> MetaVec
{
  MetaVec meta_vec;
  cxxutil::CSVRow csv_row;
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

auto read_scan_list_boreas(
  const fs::path & list_fn,
  const double & skip_len)
  -> MetaVec
{
  MetaVec meta_vec;
  cxxutil::CSVRow csv_row;
  constexpr size_t N_COL = 13UL;

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
    const auto path = std::string(csv_row[0]) + ".bin";
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

static
float getFloatFromByteArray(
  char *byteArray, uint index)
{
  return *((float *)(byteArray + index));
}

auto read_boreas_scan(
  const std::string & scan_fn,
  const double & ground_z,
  const double & sphere)
  ->pcl::PointCloud<pcl::PointXYZ>
{
  std::fstream f_bin(scan_fn, std::ios::in | std::ios::binary);
  std::vector<char> buffer(std::istreambuf_iterator<char>(f_bin), {});
  uint float_offset = 4;
  uint fields = 6; // x, y, z, i, r, t
  uint point_step = float_offset * fields;
  uint N = floor(buffer.size() / point_step);

  pcl::PointCloud<pcl::PointXYZ> cloud;

#pragma omp parallel
  for (uint i = 0; i < N; ++i)
  {
    pcl::PointXYZ point;
    uint bufpos = i * point_step;
    point.x = getFloatFromByteArray(buffer.data(), bufpos + 0 * float_offset);
    point.y = getFloatFromByteArray(buffer.data(), bufpos + 1 * float_offset);
    point.z = getFloatFromByteArray(buffer.data(), bufpos + 2 * float_offset);

    auto dist = std::sqrt(point.x * point.x
                        + point.y * point.y
                        + point.z * point.z);
    if (point.z > ground_z && dist < sphere)
    {
      cloud.push_back(point);
    }
  }

  // f_bin.seekg(0, std::ios::end);
  // const size_t num_elements = f_bin.tellg() / sizeof(float);
  // std::vector<float> buf(num_elements);
  // f_bin.seekg(0, std::ios::beg);
  // f_bin.read(
  //   reinterpret_cast<char *>(
  //     &buf[0]),
  //     num_elements * sizeof(float));
  // f_bin.close();

  // pcl::PointCloud<pcl::PointXYZI> cloud;
  // for (std::size_t i = 0; i < buf.size(); i += 6)
  // { // x, y, z, i, r, t
  //   pcl::PointXYZI point;
  //   point.x = buf[i];
  //   point.y = buf[i + 1];
  //   point.z = buf[i + 2];
  //   point.intensity = buf[i + 3];
  //   if (point.z > ground_z)
  //   {
  //     cloud.push_back(point);
  //   }
  // }
  return cloud;
}

auto readCloudXyz64(
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
    Eigen::Vector3d pv = ptXyzToVec3d(point);
    point = vec3dToPtXyz(pv);
    cloud.push_back(point);
  }
  return cloud;
}

void writeCloudXyz64(
  const std::string & scan_fn,
  const pcl::PointCloud<pcl::PointXYZ> & cloud)
{
  std::ofstream f_scan(scan_fn, std::ios::out | std::ios::binary);
  for (const auto & point : cloud.points)
  {
    f_scan.write(reinterpret_cast<const char *>(&point.x), sizeof(double));
    f_scan.write(reinterpret_cast<const char *>(&point.y), sizeof(double));
    f_scan.write(reinterpret_cast<const char *>(&point.z), sizeof(double));
  }
  f_scan.close();
}

void write_cloud_xyzi_float(
  const std::string & scan_fn,
  const pcl::PointCloud<pcl::PointXYZI> & cloud)
{
  std::ofstream f_scan(scan_fn, std::ios::out | std::ios::binary);
  for (const auto & point : cloud.points)
  {
    const auto & px = point.x;
    const auto & py = point.y;
    const auto & pz = point.z;
    const auto & pi = point.intensity;
    f_scan.write(reinterpret_cast<const char *>(&px), sizeof(float));
    f_scan.write(reinterpret_cast<const char *>(&py), sizeof(float));
    f_scan.write(reinterpret_cast<const char *>(&pz), sizeof(float));
    f_scan.write(reinterpret_cast<const char *>(&pi), sizeof(float));
  }
  f_scan.close();
}

auto scalingCloud(
  const pcl::PointCloud<pcl::PointXYZ> & cloud,
  const double & scale)
  -> pcl::PointCloud<pcl::PointXYZ>
{
  pcl::PointCloud<pcl::PointXYZ> new_cloud;
  if (scale >= 1.0)
  {
    return cloud;
  }

  const auto inv_scale = 1.0 / scale;
  for (const auto & pt: cloud.points)
  {
    auto norm = std::sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
    if (norm < scale) {
      new_cloud.emplace_back(pt.x * inv_scale,
                            pt.y * inv_scale,
                            pt.z * inv_scale);
    }
  }
  if (new_cloud.size() <= 10) {
    spdlog::error("Something Wrong with scaling: reduced size: {}", new_cloud.size());
    exit(1);
  }
  else {
    // spdlog::info("Reduced to # {}", new_cloud.size());
  }

  // Seed the random number generator
  std::random_device rd;
  std::mt19937 generator(rd());
  int min = 0;
  int max = static_cast<int>(new_cloud.size()) - 1;
  std::uniform_int_distribution<int> distribution(min, max);
  int numSamples = 4096 - static_cast<int>(new_cloud.size());
  for (int i = 0; i < numSamples; ++i)
  {
    int randi = distribution(generator);
    new_cloud.push_back(new_cloud.points[randi]);
  }

  if (new_cloud.size() != 4096UL)
  {
    spdlog::error("Plz set point size to 4096");
    exit(1);
  }

  return new_cloud;
}

}; // namespace pcpr
