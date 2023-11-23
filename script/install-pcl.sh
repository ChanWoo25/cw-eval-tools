#! /bin/bash

# Dependencies
apt update
apt install -y \
  build-essential cmake git ninja-build \
  libeigen3-dev libflann-dev libboost-all-dev

export PCL_VERSION=1.13.1
git clone https://github.com/PointCloudLibrary/pcl.git -b pcl-${PCL_VERSION} --single-branch
cd pcl

cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build install
echo "export PCL_VERSION=${PCL_VERSION}" >> ~/.bashrc
source ~/.bashrc
