// Copyright (c) 2023 Chanwoo Lee, All Rights Reserved.
// Authors: Chanwoo Lee
// Contacts: leechanwoo25@hanyang.ac.kr

#include <bits/stdint-uintn.h>
#include <cwcloud/CloudVisualizer.hpp>
#include <iostream>
#include <memory>
#include <pcl/impl/point_types.hpp>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <spdlog/fmt/bundled/format.h>
#include <spdlog/spdlog.h>

namespace cwcloud {

CloudVisualizer::CloudVisualizer(
  const std::string & root_dir,
  const int nrows,
  const int ncols)
  : nrows_(nrows),
    ncols_(ncols),
    key_sym_("")
{
  root_dir_ = root_dir;
  viewer_ = std::make_shared<pcl::visualization::PCLVisualizer>();
  viewport_ids_.resize(nrows * ncols);
  if (nrows == 1 && ncols == 1)
  {
    viewer_->createViewPort(0.0, 0.0, 1.0, 1.0, viewport_ids_[0]);
    viewer_->setBackgroundColor(0.0, 0.0, 0.0, viewport_ids_[0]);
    viewer_->addCoordinateSystem(0.05, "reference", viewport_ids_[0]);
    viewer_->setCameraPosition(
      0.559224, -2.15324, 1.97632,
      -0.274383, 0.614747, 0.739459,
      viewport_ids_[0]);
  }
  else if (nrows == 1 && ncols == 2)
  {
    viewer_->createViewPort(0.0, 0.0, 0.5, 1.0, viewport_ids_[0]);
    viewer_->setBackgroundColor (0.0, 0.0, 0.0, viewport_ids_[0]);
    viewer_->addText("Radius: 0.01", 10, 10, "v1 text", viewport_ids_[0]);
    viewer_->addCoordinateSystem (1.0, "reference", viewport_ids_[0]);
    viewer_->setCameraPosition(135.0, 45.0, 70.0, -0.335434, -0.310508, 0.889421, viewport_ids_[0]);
    // pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    // viewer_->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud1", viewport_ids_[0]);

    viewer_->createViewPort(0.5, 0.0, 1.0, 1.0, viewport_ids_[1]);
    viewer_->setBackgroundColor (0.0, 0.0, 0.0, viewport_ids_[1]);
    viewer_->addText("Radius: 0.1", 10, 10, "v2 text", viewport_ids_[1]);
    viewer_->addCoordinateSystem (1.0, "reference", viewport_ids_[1]);
    viewer_->setCameraPosition(135.0, 45.0, 70.0, -0.335434, -0.310508, 0.889421, viewport_ids_[1]);
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color(cloud, 0, 255, 0);
    // viewer_->addPointCloud<pcl::PointXYZRGB> (cloud, single_color, "sample cloud2", viewport_ids_[1]);
  }
  else if (nrows == 2 && ncols == 3)
  {
    for (int row = 0; row < nrows; ++row)
    {
      for (int col = 0; col < ncols; ++col)
      {
        const int vid = row * ncols + col;
        const double xmin = 0.3334 * static_cast<double>(col);
        const double ymin = 0.5 * static_cast<double>(row);
        const double xmax = std::min(1.0, 0.3334 * static_cast<double>(col+1));
        const double ymax = std::min(1.0, 0.5 * static_cast<double>(row+1));
        viewer_->createViewPort(xmin, ymin, xmax, ymax, viewport_ids_[vid]);
        spdlog::info("viewport id {}: {}", vid, viewport_ids_[vid]);
        viewer_->setBackgroundColor (0.0, 0.0, 0.0, viewport_ids_[vid]);
        viewer_->addCoordinateSystem (0.05,
          fmt::format("reference-v{}", vid),
          viewport_ids_[vid]);
        viewer_->setCameraPosition(
          0.559224, -2.15324, 1.97632,
          -0.274383, 0.614747, 0.739459,
          viewport_ids_[vid]);
      }
    }
    viewer_->addText("Query"   , 10, 10, "v0 text", viewport_ids_[0]);
    viewer_->addText("Positive", 10, 10, "v1 text", viewport_ids_[1]);
    viewer_->addText("Negative", 10, 10, "v2 text", viewport_ids_[2]);
    viewer_->addText("Negative", 10, 10, "v3 text", viewport_ids_[3]);
    viewer_->addText("Negative", 10, 10, "v4 text", viewport_ids_[4]);
    viewer_->addText("Negative", 10, 10, "v5 text", viewport_ids_[5]);
  }

  // auto vptr = viewer_.get();
  // viewer_->initCameraParameters ();
  viewer_->setShowFPS(true);
  viewer_->registerKeyboardCallback(
    [&](const pcl::visualization::KeyboardEvent& event)
    {
      /* Uncomment when you want to check pressed key's sym & code */
      // if (event.keyDown())
      // {
      //   spdlog::info("Key: {}/{}, Pressed: {}",
      //     event.getKeySym(),
      //     event.getKeyCode(),
      //     event.keyDown());
      // }

      if ( (   event.getKeySym() == "Right"
            || event.getKeySym() == "Left"
            || event.getKeySym() == "Up"
            || event.getKeySym() == "Down")
          && event.keyDown())
      {
        this->skip_flag = true;
        this->key_sym_ = event.getKeySym();
      }
    }
  );
}

void CloudVisualizer::run()
{
  while (!viewer_->wasStopped())
  {
    viewer_->spinOnce(10);
    std::this_thread::sleep_for(10ms);
    if (this->skip_flag)
    {
      this->skip_flag = false;
      break;
    }
  }
}

void CloudVisualizer::setCloudXYZ(
  pcl::PointCloud<pcl::PointXYZ> cloud,
  const int & nrow,
  const int & ncol,
  const std::string & _cloud_id)
{
  if (   !(0 <= nrow && nrow < nrows_)
      || !(0 <= ncol && ncol < ncols_))
  {
    spdlog::error("Out of range {}x{} | nrows:{}, ncols{}", nrow, ncol, nrows_, ncols_);
  }

  const int vid = nrow * ncols_ + ncol;

  const std::string cloud_id
    = _cloud_id.empty()
      ? fmt::format("cloud-xyz-v{}", vid)
      : _cloud_id;
  // spdlog::info("cloud_id: {}", cloud_id);

  using pcl::visualization::PointCloudColorHandlerGenericField;
  if (cloud_ids_.find(cloud_id) == cloud_ids_.end())
  {
    cloud_ids_[cloud_id] = true;
    auto color_handler
      = PointCloudColorHandlerGenericField<pcl::PointXYZ>(
          cloud.makeShared(), "z");
    // pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> color_handler(cloud, "z");
    viewer_->addPointCloud<pcl::PointXYZ>(cloud.makeShared(), color_handler, cloud_id, viewport_ids_[vid]);
    viewer_->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
      2, cloud_id, viewport_ids_[vid]);
  }
  else
  {
    auto color_handler
      = PointCloudColorHandlerGenericField<pcl::PointXYZ>(
          cloud.makeShared(), "z");
    viewer_->updatePointCloud<pcl::PointXYZ>(cloud.makeShared(), color_handler, cloud_id);
  }
}


void CloudVisualizer::setCloud(
  pcl::PointCloud<pcl::PointXYZI> cloud,
  const std::string & cloud_id,
  const int vid)
{
  using pcl::visualization::PointCloudColorHandlerGenericField;
  if (cloud_ids_.find(cloud_id) == cloud_ids_.end())
  {
    cloud_ids_[cloud_id] = true;
    auto color_handler
      = PointCloudColorHandlerGenericField<pcl::PointXYZI>(
          cloud.makeShared(), "z");
    // pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> color_handler(cloud, "z");
    viewer_->addPointCloud<pcl::PointXYZI>(cloud.makeShared(), color_handler, cloud_id, viewport_ids_[vid]);
    viewer_->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
      1, cloud_id, viewport_ids_[vid]);
  }
  else
  {
    auto color_handler
      = PointCloudColorHandlerGenericField<pcl::PointXYZI>(
          cloud.makeShared(), "z");
    viewer_->updatePointCloud<pcl::PointXYZI>(cloud.makeShared(), color_handler, cloud_id);
  }
}

void CloudVisualizer::setGroundInformCloud(
  pcl::PointCloud<pcl::PointXYZI> cloud_ground,
  pcl::PointCloud<pcl::PointXYZI> cloud_inform,
  const std::string & cloud_id,
  const int vid)
{
  using pcl::visualization::PointCloudColorHandlerGenericField;
  const auto cloud_id_ground = cloud_id + "_ground";
  const auto cloud_id_inform = cloud_id + "_inform";

  if (cloud_ids_.find(cloud_id) == cloud_ids_.end())
  {
    cloud_ids_[cloud_id] = true;

    auto color_handler_ground
      = pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>(
          cloud_ground.makeShared(), 0.0, 0.0, 255.0);
    viewer_->addPointCloud<pcl::PointXYZI>(
      cloud_ground.makeShared(),
      color_handler_ground,
      cloud_id_ground,
      viewport_ids_[vid]);
    viewer_->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
      1, cloud_id_ground,
      viewport_ids_[vid]);

    auto color_handler_inform
      = PointCloudColorHandlerGenericField<pcl::PointXYZI>(
          cloud_inform.makeShared(), "z");
    viewer_->addPointCloud<pcl::PointXYZI>(
      cloud_inform.makeShared(),
      color_handler_inform,
      cloud_id_inform,
      viewport_ids_[vid]);
    viewer_->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
      1, cloud_id_inform,
      viewport_ids_[vid]);
  }
  else
  {
    auto color_handler_ground
      = pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>(
          cloud_ground.makeShared(), 0.0, 0.0, 255.0);
    viewer_->updatePointCloud<pcl::PointXYZI>(
      cloud_ground.makeShared(),
      color_handler_ground,
      cloud_id_ground);
    auto color_handler_inform
      = PointCloudColorHandlerGenericField<pcl::PointXYZI>(
          cloud_inform.makeShared(), "z");
    viewer_->updatePointCloud<pcl::PointXYZI>(
      cloud_inform.makeShared(),
      color_handler_inform,
      cloud_id_inform);
  }
}


} // namespace cwcloud
