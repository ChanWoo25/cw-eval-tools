cmake_minimum_required(VERSION 3.11)

set(PACKAGE_NAME CXX_TOOLS)
set(PACKAGE_VERSION 1.0.0)

project(${PACKAGE_NAME} VERSION ${PACKAGE_VERSION} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "" FORCE)
set(CMAKE_BUILD_TYPE "Release")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

set(${PACKAGE_NAME}_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(${PACKAGE_NAME}_SOURCE_PATH "${${PACKAGE_NAME}_PATH}/src")
set(${PACKAGE_NAME}_INCLUDE_PATH "${${PACKAGE_NAME}_PATH}/include")
set(${PACKAGE_NAME}_TEST_PATH "${${PACKAGE_NAME}_PATH}/test")
set(${PACKAGE_NAME}_RESOURCE_PATH "${${PACKAGE_NAME}_PATH}/resource")
set(${PACKAGE_NAME}_MODULE_PATH "${${PACKAGE_NAME}_PATH}/module")
set(${PACKAGE_NAME}_THIRDPARTY_PATH "${${PACKAGE_NAME}_PATH}/thirdparty")

message(STATUS "PACKAGE_NAME: ${PACKAGE_NAME}")
message(STATUS "PACKAGE_VERSION: ${PACKAGE_VERSION}")
message(STATUS "${PACKAGE_NAME}_PATH: ${${PACKAGE_NAME}_PATH}")
message(STATUS "${PACKAGE_NAME}_SOURCE_PATH: ${${PACKAGE_NAME}_SOURCE_PATH}")
message(STATUS "${PACKAGE_NAME}_INCLUDE_PATH: ${${PACKAGE_NAME}_INCLUDE_PATH}")
message(STATUS "${PACKAGE_NAME}_TEST_PATH: ${${PACKAGE_NAME}_TEST_PATH}")
message(STATUS "${PACKAGE_NAME}_RESOURCE_PATH: ${${PACKAGE_NAME}_RESOURCE_PATH}")
message(STATUS "${PACKAGE_NAME}_MODULE_PATH: ${${PACKAGE_NAME}_MODULE_PATH}")
message(STATUS "${PACKAGE_NAME}_THIRDPARTY_PATH: ${${PACKAGE_NAME}_THIRDPARTY_PATH}\n")

## Find OpenCV
find_package(OpenCV 4.2 REQUIRED)
if(OpenCV_FOUND)
  message(STATUS "OPENCV_VERSION: ${OpenCV_VERSION}")
  message(STATUS "OPENCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
  # message(STATUS "Found OpenCV libraries is ${OpenCV_LIBS}")
endif()

## Find argparse
find_package(argparse 3.0.0 REQUIRED)
if (argparse_FOUND)
  message(STATUS "Found argparse.")
  message(STATUS "* Version: ${argparse_VERSION}\n")
  # message(STATUS "* Usage: ")
  # message(STATUS "* - find_package(argparse {VERSION} REQUIRED)")
  # message(STATUS "* - target_include_directories(target argparse::argparse)")
endif()

find_package(PkgConfig REQUIRED)

# pkg_check_modules(GLOG REQUIRED libglog)
# if (GLOG_FOUND)
#   message(STATUS "Found glog.")
#   message(STATUS "* Version: ${GLOG_VERSION}")
#   message(STATUS "* Libs: ${GLOG_LIBRARIES}")
#   message(STATUS "* Ex: target_link_libraries(example PRIVATE glog::glog)")
# endif()

# pkg_check_modules(GFLAGS REQUIRED gflags)
# if (GFLAGS_FOUND)
#   message(STATUS "Found gflags.")
#   message(STATUS "GFLAGS_VERSION: ${GFLAGS_VERSION}")
#   message(STATUS "* Version: ${GFLAGS_VERSION}")
#   message(STATUS "* Libs: ${GFLAGS_LIBRARIES}")
# endif()

pkg_check_modules(EIGEN3 REQUIRED eigen3)
if (EIGEN3_FOUND)
  message(STATUS "Found eigen3.")
  message(STATUS "* Version: ${EIGEN3_VERSION}")
  message(STATUS "* Header: ${EIGEN3_INCLUDE_DIRS}")
  message(STATUS "* Tip: Just target_include_directories(example $<EIGEN3_INCLUDE_DIRS> )")
endif()

pkg_check_modules(FLANN REQUIRED flann)
if (FLANN_FOUND)
  message(STATUS "Found flann.")
  message(STATUS "* Version: ${FLANN_VERSION}")
  message(STATUS "* Header: ${FLANN_INCLUDE_DIRS}")
  message(STATUS "* Libs: ${FLANN_LIBRARIES}")
  message(STATUS "* Tip: Just target_include_directories(example $<FLANN_INCLUDE_DIRS> )")
endif()
#     add_executable(foo foo.cc)
#     target_link_libraries(foo FLANN::FLANN)


include("cmake/find_spdlog.cmake")
include("cmake/find_pcl.cmake")

add_subdirectory(${${PACKAGE_NAME}_THIRDPARTY_PATH})
add_subdirectory(${${PACKAGE_NAME}_MODULE_PATH})

# add_executable(MAIN src/main.cpp)
# target_link_libraries(MAIN
#   glog
#   gflags
#   fmt::fmt
#   ${SPDLOG_LIBS}
# )
# target_compile_options(MAIN PRIVATE
#   -Wall -Wextra -Wpedantic -Werror
# )
# target_include_directories(MAIN PRIVATE
#   ${SPDLOG_INCLUDE_PATH}
# )

############################################
# message(${${PACKAGE_NAME}_PATH}/build/module/cwcloud/libCWCLOUD.a)
# message(STATUS "SPDLOG_INCLUDE_PATH: ${SPDLOG_INCLUDE_PATH}")
# message(STATUS "SPDLOG_LIBS: ${SPDLOG_LIBS}")
# message(STATUS "PCL_LIBS: ${PCL_LIBS}")

# add_executable(parse_point_cloud
#   src/parse_point_cloud.cpp
#   src/pcpr.cpp)
# target_link_libraries(parse_point_cloud
#   CWCLOUD
#   CXXUTIL
#   ${SPDLOG_LIBS}
#   "${PCL_LIBS}"
# )
# target_include_directories(parse_point_cloud PUBLIC
#   ${${PACKAGE_NAME}_INCLUDE_PATH}
#   ${SPDLOG_INCLUDE_PATH}
#   ${PCL_INCLUDE_PATH}
#   ${EIGEN3_INCLUDE_DIRS}
# )
# target_compile_options(parse_point_cloud PRIVATE
#   -Wall -Wextra -Wpedantic -Werror
# )

# add_executable(debug_pcpr
#   src/debug_pcpr.cpp
#   src/pcpr.cpp)
# target_link_libraries(debug_pcpr
#   CWCLOUD
#   CXXUTIL
#   ${SPDLOG_LIBS}
#   "${PCL_LIBS}"
#   argparse::argparse
# )
# target_include_directories(debug_pcpr PUBLIC
#   ${${PACKAGE_NAME}_INCLUDE_PATH}
#   ${SPDLOG_INCLUDE_PATH}
#   ${PCL_INCLUDE_PATH}
#   ${EIGEN3_INCLUDE_DIRS}
#   ${ARGPARSE_INCLUDE_DIRS}
# )
# target_compile_options(debug_pcpr PRIVATE
#   -Wall -Wextra -Wpedantic -Werror
# )

# add_executable(explore_cs_campus
#   src/explore_cs_campus.cpp
#   src/pcpr.cpp)
# target_link_libraries(explore_cs_campus
#   CWCLOUD
#   CXXUTIL
#   ${SPDLOG_LIBS}
#   "${PCL_LIBS}"
# )
# target_include_directories(explore_cs_campus PUBLIC
#   ${${PACKAGE_NAME}_INCLUDE_PATH}
#   ${SPDLOG_INCLUDE_PATH}
#   ${PCL_INCLUDE_PATH}
#   ${EIGEN3_INCLUDE_DIRS}
# )
# target_compile_options(explore_cs_campus PRIVATE
#   -Wall -Wextra -Wpedantic -Werror
# )


# fetch latest argparse
# include(FetchContent)
# FetchContent_Declare(
#     argparse
#     GIT_REPOSITORY https://github.com/p-ranav/argparse.git
# )
# FetchContent_MakeAvailable(argparse)


# add_executable(save_new_catalog
#   src/save_new_catalog.cpp
#   src/pcpr.cpp)
# target_link_libraries(save_new_catalog
#   CWCLOUD
#   CXXUTIL
#   ${SPDLOG_LIBS}
#   "${PCL_LIBS}"
#   argparse::argparse
# )
# target_include_directories(save_new_catalog PUBLIC
#   ${${PACKAGE_NAME}_INCLUDE_PATH}
#   ${SPDLOG_INCLUDE_PATH}
#   ${PCL_INCLUDE_PATH}
#   ${EIGEN3_INCLUDE_DIRS}
#   ${ARGPARSE_INCLUDE_DIRS}
# )
# target_compile_options(save_new_catalog PRIVATE
#   -Wall -Wextra -Wpedantic -Werror
# )


add_executable(main
  src/pcpr.cpp
  src/preprocess.cpp
  main.cpp)
target_link_libraries(main
  CWCLOUD
  CXXUTIL
  ${SPDLOG_LIBS}
  "${PCL_LIBS}"
  argparse::argparse
)
target_include_directories(main PUBLIC
  ${${PACKAGE_NAME}_INCLUDE_PATH}
  ${SPDLOG_INCLUDE_PATH}
  ${PCL_INCLUDE_PATH}
  ${EIGEN3_INCLUDE_DIRS}
  ${ARGPARSE_INCLUDE_DIRS}
)
target_compile_options(main PRIVATE
  -Wall -Wextra -Wpedantic -Werror
)


