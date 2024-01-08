cmake_minimum_required(VERSION 3.11)

set(PACKAGE_NAME coding_test)
set(PACKAGE_VERSION 1.0.0)

project(${PACKAGE_NAME} VERSION ${PACKAGE_VERSION} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "" FORCE)
set(CMAKE_BUILD_TYPE "Release")

set(${PACKAGE_NAME}_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(${PACKAGE_NAME}_SOURCE_PATH "${${PACKAGE_NAME}_PATH}/src")
set(${PACKAGE_NAME}_INCLUDE_PATH "${${PACKAGE_NAME}_PATH}/include")
set(${PACKAGE_NAME}_TEST_PATH "${${PACKAGE_NAME}_PATH}/test")
set(${PACKAGE_NAME}_RESOURCE_PATH "${${PACKAGE_NAME}_PATH}/resource")
set(${PACKAGE_NAME}_MODULE_PATH "${${PACKAGE_NAME}_PATH}/module")
set(${PACKAGE_NAME}_THIRDPARTY_PATH "${${PACKAGE_NAME}_PATH}/thirdparty")

## Find argparse
find_package(argparse 3.0.0 REQUIRED)
if (argparse_FOUND)
  message(STATUS "Found argparse.")
  message(STATUS "* Version: ${argparse_VERSION}\n")
  # message(STATUS "* Usage: ")
  # message(STATUS "* - find_package(argparse {VERSION} REQUIRED)")
  # message(STATUS "* - target_include_directories(target argparse::argparse)")
endif()

## Find yaml-cpp
find_package(yaml-cpp REQUIRED)
if (YAML_CPP_FOUND)
  message(STATUS "Found yaml-cpp.")
  message(STATUS "* Version: ${YAML_CPP_VERSION}\n")
  message(STATUS "* YAML_CPP_LIBRARIES: ${YAML_CPP_LIBRARIES}")
  message(STATUS "* YAML_CPP_INCLUDE_DIRS: ${YAML_CPP_INCLUDE_DIRS}")
  # message(STATUS "* Usage: ")
  # message(STATUS "* - find_package(argparse {VERSION} REQUIRED)")
  # message(STATUS "* - target_include_directories(target argparse::argparse)")
endif()

find_package(PkgConfig REQUIRED)

pkg_check_modules(EIGEN3 REQUIRED eigen3)
if (EIGEN3_FOUND)
  message(STATUS "Found eigen3.")
  message(STATUS "* Version: ${EIGEN3_VERSION}")
  message(STATUS "* Header: ${EIGEN3_INCLUDE_DIRS}")
  message(STATUS "* Tip: Just target_include_directories(example $<EIGEN3_INCLUDE_DIRS> )")
endif()

add_subdirectory(${${PACKAGE_NAME}_THIRDPARTY_PATH})

include("cmake/find_spdlog.cmake")

###############
# Executables #
###############
add_executable(${PACKAGE_NAME}
  src/archive.cpp
  main.cc)
target_link_libraries(${PACKAGE_NAME}
  ${SPDLOG_LIBS}
  ${YAML_CPP_LIBRARIES}
)
target_include_directories(${PACKAGE_NAME} PUBLIC
  ${${PACKAGE_NAME}_INCLUDE_PATH}
  ${SPDLOG_INCLUDE_PATH}
  ${YAML_CPP_INCLUDE_DIRS}
)
target_compile_options(${PACKAGE_NAME} PRIVATE
  -Wall -Wextra -Wpedantic -Werror
)
