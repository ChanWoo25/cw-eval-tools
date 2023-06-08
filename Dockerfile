# Cuda Toolkit version is following pytorch 1.8.2 LTS Version.
FROM nvidia/cudagl:11.1.1-devel-ubuntu20.04

# Declare variables
ARG DEBIAN_FRONTEND=noninteractive \
    EIGEN_VERSION=3.4.0 \
    OPENCV_VERSION=3.4.19

# For fixing broken cuda apt package list
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub &&\
    apt update && apt dist-upgrade -y &&\
    apt autoremove -y && apt clean

RUN apt install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
      build-essential cmake wget curl unzip git vim openssh-server openssh-client \
      libgoogle-glog-dev libgflags-dev vlc pkg-config \
      python3-dev python3-pip python3-setuptools \
      gfortran libprotobuf-dev protobuf-compiler &&\
    apt autoremove -y && apt clean

RUN apt install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
      libblas-dev liblapack-dev liblapacke-dev \
      libfmt-dev tzdata dirmngr gnupg2 \
      libjpeg-dev libpng-dev libtiff-dev libtbb-dev libgtk2.0-dev \
      ffmpeg libavcodec-dev libavformat-dev libswscale-dev \
      libxvidcore-dev libx264-dev libxine2-dev \
      libatlas-base-dev libsuitesparse-dev \
      mesa-utils libgl1-mesa-glx  \
      freeglut3-dev libglfw3-dev libglew-dev  &&\
    apt autoremove -y && apt clean

# Install Eigen
RUN git clone -b ${EIGEN_VERSION} \
              --single-branch \
              https://gitlab.com/libeigen/eigen.git &&\
    cd eigen &&\
    mkdir build && cd build &&\
    cmake .. && make -j8 install && ldconfig && \
    cd && rm -rf eigen

# Install OpenCV
RUN cd && git config --global http.sslverify false &&\
    git clone https://github.com/opencv/opencv.git \
      --branch ${OPENCV_VERSION} --single-branch &&\
    cd opencv && mkdir build && cd build &&\
    cmake \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      .. &&\
    make -j8 install && ldconfig &&\
    cd && rm -rf ~/opencv*

RUN apt update &&\
 apt install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
  freeglut3-dev libglfw3-dev libglew-dev &&\
 apt-get clean && apt-get autoremove -y &&\
 rm -rf /var/lib/apt/lists/*

# Install Python 3 & Pytorch
RUN pip3 install -U pip &&\
    pip3 install -U pillow wheel numpy pandas seaborn scipy matplotlib nbformat &&\
    pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

RUN echo "export LIBGL_ALWAYS_INDIRECT=1" >> /root/.bashrc &&\
    echo "check_certificate = off" >> ~/.wgetrc

# RUN cd &&\
#  git clone https://github.com/strasdat/Sophus.git --single-branch -b 1.22.10 &&\
#  cd Sophus && mkdir build && cd build &&\
#  cmake .. && make install &&\
#  cd && rm -rf Sophus

# # Install Open3D
# RUN export OPEN3D_VERSION=v0.16.1 &&\
#  cd ~ && apt update &&\
#  apt install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
#   # Open3D
#   xorg-dev libxcb-shm0 libglu1-mesa-dev \
#   # Filament build-from-source
#   clang-7 libc++-7-dev libc++abi-7-dev \
#   libsdl2-dev ninja-build libxi-dev \
#   # Headless rendering
#   libosmesa6-dev &&\
#  git config --global http.sslverify false &&\
#  git clone https://github.com/isl-org/Open3D --branch ${OPEN3D_VERSION} --single-branch &&\
#  cd Open3D && mkdir build && cd build &&\
#  cmake \
#   -DBUILD_PYTHON_MODULE=ON \
#   -DPython3_ROOT=/usr/bin/python3 \
#   -DCMAKE_INSTALL_PREFIX=/usr/local/open3d_install \
#   -DBUILD_SHARED_LIBS=ON \
#   .. &&\
#  make install -j 8 &&\
#  apt-get clean && apt-get autoremove -y &&\
#  rm -rf /var/lib/apt/lists/*
#  # After install this, refer here "https://github.com/isl-org/open3d-cmake-find-package" to link open3d libs with your project
