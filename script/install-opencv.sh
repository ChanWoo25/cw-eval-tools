#! /bin/bash
# Author: Chanwoo Lee (https://github.com/ChanWoo25)
#

# TODO
# 1. According to OpenCV Version, install differenct dependencies.

function install_opencv_in_local() {
  local target_version=$1
  echo "Install OpenCV ${target_version} in local environment..."

  DEBIAN_FRONTEND=noninteractive
  # Install Dependencies
  sudo apt-get update
  sudo apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        cmake build-essential git pkg-config \
        gfortran libprotobuf-dev protobuf-compiler libomp-dev \
        libblas-dev libopenblas-dev liblapack-dev liblapacke-dev \
        libjpeg-dev libpng-dev libtiff-dev libtbb-dev libgtk2.0-dev libhdf5-dev \
        ffmpeg libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev \
        libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev libgl1-mesa-dev
  sudo apt-get clean
  sudo apt-get autoremove -y

  # Install Ceres
  export CERES_VERSION=1.14.0
  cd ~ &&
  git clone https://ceres-solver.googlesource.com/ceres-solver &&
  cd ceres-solver &&
  git checkout ${CERES_VERSION} &&
  mkdir build &&
  cd build &&
  cmake .. &&
  make -j8 install && ldconfig && cd ~ &&
  rm -rf ceres-solver

  # Install OpenCV
  # Ceres 2.1.0 has some new function that build issue occurs when building sfm
  cd ~
  # git config --global http.sslverify false &&
  git clone https://github.com/opencv/opencv.git --branch ${target_version} --single-branch
  git clone https://github.com/opencv/opencv_contrib.git --branch ${target_version} --single-branch

  cd opencv
  mkdir build && cd build

  # atmost packages will be detected automatically
  cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
        ..
  sudo make -j8 install
  sudo ldconfig
  rm -rf ~/opencv*
}

function install_opencv_in_docker() {
  local target_version=$1
  echo "Install OpenCV ${target_version} in docker container ..."

  DEBIAN_FRONTEND=noninteractive
  # Install Dependencies
  apt-get update
  apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    cmake build-essential git \
    gfortran libprotobuf-dev protobuf-compiler \
    libblas-dev libopenblas-dev liblapack-dev liblapacke-dev \
    libjpeg-dev libpng-dev libtiff-dev libtbb-dev libgtk2.0-dev pkg-config \
    ffmpeg libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev \
    libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev libgl1-mesa-dev
  apt-get clean
  apt-get autoremove -y

  # Install Ceres
  export CERES_VERSION=1.14.0
  cd ~ &&
  git clone https://ceres-solver.googlesource.com/ceres-solver &&
  cd ceres-solver &&
  git checkout ${CERES_VERSION} &&
  mkdir build &&
  cd build &&
  cmake .. &&
  make -j8 install && ldconfig && cd ~ &&
  rm -rf ceres-solver

  # Install OpenCV
  # Ceres 2.1.0 has some new function that build issue occurs when building sfm
  cd ~
  # git config --global http.sslverify false &&
  git clone https://github.com/opencv/opencv.git --branch ${target_version} --single-branch
  git clone https://github.com/opencv/opencv_contrib.git --branch ${target_version} --single-branch

  cd opencv
  mkdir build && cd build

  # atmost packages will be detected automatically
  cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
        ..
  make -j8 install
  ldconfig
  rm -rf ~/opencv*
}

# Arguments check
if [ $# -lt 2 ]; then
  echo "Example: install-opencv.sh <local or docker> <version>"
  exit 1
fi

MODE=$1
TARGET_VERSION=$2

cd ~

case $MODE in
  "local")
    install_opencv_in_local "$TARGET_VERSION"
    ;;
  "docker")
    install_opencv_in_docker "$TARGET_VERSION"
    ;;
  *)
    echo "Wrong function name !!"
    exit 1
    ;;
esac

exit 0
