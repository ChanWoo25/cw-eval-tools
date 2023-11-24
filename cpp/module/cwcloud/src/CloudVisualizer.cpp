// Copyright (c) 2023 Chanwoo Lee, All Rights Reserved.
// Authors: Chanwoo Lee
// Contacts: leechanwoo25@hanyang.ac.kr

#include <cwcloud/CloudVisualizer.hpp>
#include <iostream>
#include <memory>
#include <pcl/impl/point_types.hpp>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <spdlog/spdlog.h>

namespace cwcloud {

CloudVisualizer::CloudVisualizer(
  const std::string & root_dir,
  const int nrows,
  const int ncols)
{
  root_dir_ = root_dir;
  viewer_ = std::make_shared<pcl::visualization::PCLVisualizer>();
  viewport_ids_.resize(nrows * ncols);
  if (nrows == 1 && ncols == 1)
  {
    viewer_->createViewPort(0.0, 0.0, 1.0, 1.0, viewport_ids_[0]);
    viewer_->setBackgroundColor (1.0, 1.0, 1.0);
    viewer_->addCoordinateSystem (1.0);
    viewer_->setCameraPosition(1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
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

  auto print_KeyCB = [&](
    const pcl::visualization::KeyboardEvent& event) //
  {
    if ((event.getKeySym() == "a" || event.getKeySym() == "A")
        && event.keyDown())
    {
      spdlog::info("Key: {}/{}, Pressed: {}",
        event.getKeySym(),
        event.getKeyCode(),
        event.keyDown());
      this->skip_flag = true;
    }
  };

  // auto vptr = viewer_.get();
  // viewer_->initCameraParameters ();
  viewer_->setShowFPS(true);
  viewer_->registerKeyboardCallback(print_KeyCB);

  // auto cloud = pcl::PointCloud<pcl::PointXYZI>().makeShared();
  // viewer_->addPointCloud<pcl::PointXYZI> (cloud, "sample cloud");
  // viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
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
