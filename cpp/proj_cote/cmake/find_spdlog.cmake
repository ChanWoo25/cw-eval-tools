
if(spdlog_FOUND)
else()
  set(SPDLOG_INSTALL_PATH ${CMAKE_BINARY_DIR}/thirdparty/spdlog/install/${CMAKE_BUILD_TYPE})
  find_package(spdlog QUIET PATHS ${SPDLOG_INSTALL_PATH}/lib/cmake/spdlog)
endif()


if(spdlog_FOUND)
  message(STATUS "Found spdlog.")
  message(STATUS "* Version: ${spdlog_VERSION}\n")
  set(SPDLOG_INCLUDE_PATH ${SPDLOG_INSTALL_PATH}/include)
  set(SPDLOG_LIBS spdlog::spdlog)
  # include_directories(${SPDLOG_INCLUDE_PATH})
elseif()
  message(STATUS "Not found spdlog.\n")
endif()
