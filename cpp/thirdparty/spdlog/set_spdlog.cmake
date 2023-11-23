message(STATUS "Start configuration for spdlog.")

set(SPDLOG_INSTALL_PATH ${CMAKE_CURRENT_BINARY_DIR}/install/${CMAKE_BUILD_TYPE})
find_package(spdlog QUIET PATHS ${SPDLOG_INSTALL_PATH}/lib/cmake/spdlog)

if(spdlog_FOUND)
  message(STATUS "Found spdlog.")
  message(STATUS "* Version: ${spdlog_VERSION}")
  message(STATUS "* Ex: target_link_libraries(example PRIVATE spdlog::spdlog)")
  # message(STATUS "* Used config file: ${spdlog_CONFIG}")
else()
  message(STATUS "Not found spdlog. Using ExternalProject.")
  ExternalProject_Add(
    SPDLOG
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.12.0
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/spdlog-prefix
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
    -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/install/${CMAKE_BUILD_TYPE}
    -DSPDLOG_BUILD_SHARED=OFF
  )
endif(spdlog_FOUND)

message(STATUS "Finish configuration for spdlog.")
