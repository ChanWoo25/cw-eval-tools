message(STATUS "Start configuration for yaml-cpp.")

set(YAML_CPP_INSTALL_PATH ${CMAKE_CURRENT_BINARY_DIR}/install/${CMAKE_BUILD_TYPE})
find_package(yaml-cpp 0.6.3 EXACT QUIET PATHS ${YAML_CPP_INSTALL_PATH}/lib/cmake/yaml-cpp)

if(yaml-cpp_FOUND)
  message(STATUS "Found yaml-cpp.")
  message(STATUS "* Version: ${yaml-cpp_VERSION}")
  message(STATUS "* Include: ${YAML_CPP_INCLUDE_DIR}")
  message(STATUS "* Libs: ${YAML_CPP_LIBRARIES}")
  # message(STATUS "* Used config file: ${yaml-cpp_CONFIG}")
else()
  message(STATUS "Not found spdlog. Using ExternalProject.")
  ExternalProject_Add(
    YAML_CPP
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG yaml-cpp-0.6.3
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/yaml-cpp-prefix
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/install/${CMAKE_BUILD_TYPE}
      -DYAML_CPP_BUILD_TESTS:BOOL=OFF
  )
endif()

message(STATUS "Finish configuration for spdlog.")
