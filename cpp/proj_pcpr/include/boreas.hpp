#ifndef BOREAS_HPP_
#define BOREAS_HPP_

#include <bits/stdint-uintn.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <iterator>
#include <filesystem>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

namespace fs = std::filesystem;

double upgrade_time = 1632182400.0;

static int64_t getStampFromPath(const std::string path)
{
  return std::stoll(fs::path(path).stem().string());
}

/*!
  \brief Decode a single Oxford Radar RobotCar Dataset radar example
  \param path path to the radar image png file
  \param timestamps [out] Timestamp for each azimuth in int64 (UNIX time) in microseconds
  \param azimuths [out] Rotation for each polar radar azimuth (radians)
  \param valid [out] Mask of whether azimuth data is an original sensor reasing or interpolated from adjacent azimuths
  \param fft_data [out] Radar power readings along each azimuth min=0, max=1
*/
void load_radar(
  std::string path,
  std::vector<int64_t> & timestamps,
  std::vector<double> & azimuths,
  cv::Mat & fft_data,
  double & resolution)
{
  uint encoder_size = 5600;
  double t = double(getStampFromPath(path)) * 1.0e-6;
  if (t > upgrade_time)
  {
    resolution = 0.04381;
  }
  else
  {
    resolution = 0.0596;
  }

  uint min_range = round(2.5 / resolution);
  cv::Mat raw_data = cv::imread(path, cv::IMREAD_GRAYSCALE);

  uint N = raw_data.rows;
  timestamps = std::vector<int64_t>(N, 0);
  azimuths = std::vector<double>(N, 0);
  uint range_bins = raw_data.cols - 11;
  fft_data = cv::Mat::zeros(N, range_bins, CV_32F);
#pragma omp parallel
  for (uint i = 0; i < N; ++i)
  {
      uint8_t * byteArray = raw_data.ptr<uint8_t>(i);
      timestamps[i] = *((int64_t *)(byteArray));
      azimuths[i] = *((uint16_t *)(byteArray + 8)) * 2 * M_PI / double(encoder_size);
      for (uint j = min_range; j < range_bins; j++)
      {
          fft_data.at<float>(i, j) = (float)*(byteArray + 11 + j) / 255.0;
      }
  }
}

static float getFloatFromByteArray(char *byteArray, uint index)
{
  return *((float *)(byteArray + index));
}

// Input is a .bin binary file.
void load_lidar(std::string path, Eigen::MatrixXd &pc)
{
  std::ifstream ifs(path, std::ios::binary);
  std::vector<char> buffer(std::istreambuf_iterator<char>(ifs), {});
  uint float_offset = 4;
  uint fields = 6; // x, y, z, i, r, t
  uint point_step = float_offset * fields;
  uint N = floor(buffer.size() / point_step);
  pc = Eigen::MatrixXd::Ones(N, fields);
#pragma omp parallel
  for (uint i = 0; i < N; ++i)
  {
    uint bufpos = i * point_step;
    for (uint j = 0; j < fields; ++j)
    {
      pc(i, j) = getFloatFromByteArray(buffer.data(), bufpos + j * float_offset);
    }
  }
  // Add offset to timestamps
  double t = double(getStampFromPath(path)) * 1.0e-6;
  pc.block(0, 5, N, 1).array() += t;
}

static double get_azimuth_index(const std::vector<double> &azimuths, double azimuth)
{
    auto lower = std::lower_bound(azimuths.begin(), azimuths.end(), azimuth);
    double closest = std::distance(azimuths.begin(), lower);
    uint M = azimuths.size();
    if (closest >= M)
        closest = M - 1;

    if (azimuths[closest] < azimuth)
    {
        double delta = 0;
        if (closest < M - 1)
        {
            if (azimuths[closest + 1] == azimuths[closest])
                delta = 0.5;
            else
                delta = (azimuth - azimuths[closest]) / (azimuths[closest + 1] - azimuths[closest]);
        }
        closest += delta;
    }
    else if (azimuths[closest] > azimuth)
    {
        double delta = 0;
        if (closest > 0)
        {
            if (azimuths[closest - 1] == azimuths[closest])
                delta = 0.5;
            else
                delta = (azimuths[closest] - azimuth) / (azimuths[closest] - azimuths[closest - 1]);
        }
        closest -= delta;
    }
    return closest;
}

/*!
   \brief Decode a single Oxford Radar RobotCar Dataset radar example
   \param polar_in Radar power readings along each azimuth
   \param azimuths_in Rotation for each polar radar azimuth (radians)
   \param cart_out [out] Cartesian radar power readings
   \param radar_resolution Resolution of the polar radar data (metres per pixel)
   \param cart_resolution Cartesian resolution (meters per pixel)
   \param cart_pixel_width Width and height of the returned square cartesian output (pixels).
   \param interpolate_crossover If true, interpolates between the end and start azimuth of the scan.
*/
void radar_polar_to_cartesian(const cv::Mat &polar_in, const std::vector<double> &azimuths_in, cv::Mat &cart_out,
                              float radar_resolution = 0.0596, float cart_resolution = 0.2384, int cart_pixel_width = 640, bool fix_wobble = true)
{

    float cart_min_range = (cart_pixel_width / 2) * cart_resolution;
    if (cart_pixel_width % 2 == 0)
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution;

    cv::Mat map_x = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);
    cv::Mat map_y = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);

#pragma omp parallel for collapse(2)
    for (int j = 0; j < map_y.cols; ++j)
    {
        for (int i = 0; i < map_y.rows; ++i)
        {
            map_y.at<float>(i, j) = -1 * cart_min_range + j * cart_resolution;
        }
    }
#pragma omp parallel for collapse(2)
    for (int i = 0; i < map_x.rows; ++i)
    {
        for (int j = 0; j < map_x.cols; ++j)
        {
            map_x.at<float>(i, j) = cart_min_range - i * cart_resolution;
        }
    }
    cv::Mat range = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);
    cv::Mat angle = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);

    uint M = azimuths_in.size();
    double azimuth_step = (azimuths_in[M - 1] - azimuths_in[0]) / (M - 1);
#pragma omp parallel for collapse(2)
    for (int i = 0; i < range.rows; ++i)
    {
        for (int j = 0; j < range.cols; ++j)
        {
            float x = map_x.at<float>(i, j);
            float y = map_y.at<float>(i, j);
            float r = (sqrt(pow(x, 2) + pow(y, 2)) - radar_resolution / 2) / radar_resolution;
            if (r < 0)
                r = 0;
            range.at<float>(i, j) = r;
            float theta = atan2f(y, x);
            if (theta < 0)
                theta += 2 * M_PI;
            if (fix_wobble and radar_resolution == 0.0596)
            { // fix wobble in CIR204-H data
                angle.at<float>(i, j) = get_azimuth_index(azimuths_in, theta);
            }
            else
            {
                angle.at<float>(i, j) = (theta - azimuths_in[0]) / azimuth_step;
            }
        }
    }
    // interpolate cross-over
    cv::Mat a0 = cv::Mat::zeros(1, polar_in.cols, CV_32F);
    cv::Mat aN_1 = cv::Mat::zeros(1, polar_in.cols, CV_32F);
    for (int j = 0; j < polar_in.cols; ++j)
    {
        a0.at<float>(0, j) = polar_in.at<float>(0, j);
        aN_1.at<float>(0, j) = polar_in.at<float>(polar_in.rows - 1, j);
    }
    cv::Mat polar = polar_in.clone();
    cv::vconcat(aN_1, polar, polar);
    cv::vconcat(polar, a0, polar);
    angle = angle + 1;
    // polar to cart warp
    cv::remap(polar, cart_out, range, angle, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
}

#endif
