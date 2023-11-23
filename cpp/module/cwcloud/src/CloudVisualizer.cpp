// Copyright (c) 2023 Chanwoo Lee, All Rights Reserved.
// Authors: Chanwoo Lee
// Contacts: leechanwoo25@hanyang.ac.kr

#include <cwcloud/CloudVisualizer.hpp>
#include <iostream>
#include <memory>

namespace cwcloud {

CloudVisualizer::CloudVisualizer(
  const std::string & root_dir)
{
  root_dir_ = root_dir;
  viewer_ = std::make_shared<pcl::visualization::PCLVisualizer>();
  viewer_->setBackgroundColor (1.0, 1.0, 1.0);
  viewer_->addCoordinateSystem (1.0);
  viewer_->initCameraParameters ();
  viewer_->setShowFPS(true);
  viewer_->setCameraPosition(1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

  auto print_KeyCB = [](
    const pcl::visualization::KeyboardEvent& event,
    void * ) //
  {
    // if (event.getKeySym() == "s" && event.keyDown())
    // {
    //   std::cout << "Key 's' pressed. Do something here." << std::endl;
    // }
    spdlog::info("Key: {}/{}, Pressed: {}",
      event.getKeySym(),
      event.getKeyCode(),
      event.keyDown());
  };
  // // void keyboardEventOccurred(
  // //   )
  // // {
  // //   spdlog::info("Key: {}/{}, Pressed: {}", event.getKeySym(), event.getKeyCode(), event.keyDown());
  // //   // if (event.getKeySym() == "s" && event.keyDown()) {
  // //   //   std::cout << "Key 's' pressed. Do something here." << std::endl;
  // //   //   // Add your custom logic for the 's' key press
  // //   // }
  // // }
  viewer_->registerKeyboardCallback(print_KeyCB, viewer_.get());

  auto cloud = pcl::PointCloud<pcl::PointXYZI>().makeShared();
  viewer_->addPointCloud<pcl::PointXYZI> (cloud, "sample cloud");
  viewer_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
}

void CloudVisualizer::run()
{
  while (!viewer_->wasStopped())
  {
    viewer_->spinOnce(100);
    std::this_thread::sleep_for(100ms);
  }
}

void CloudVisualizer::setCloud(
  pcl::PointCloud<pcl::PointXYZI> cloud,
  const std::string & cloud_id)
{
  // std::unique_lock<std::shared_mutex> lock(smtx_cloud_);

  viewer_->addPointCloud<pcl::PointXYZI>(cloud.makeShared(), cloud_id);
  viewer_->setPointCloudRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
    1, cloud_id);

  // viewer_->updatePointCloud<pcl::PointXYZI>(cloud.makeShared(), cloud_id);
}

} // namespace cwcloud
