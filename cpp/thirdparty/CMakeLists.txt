message(STATUS "Start Building thirdparties.")

file(GLOB ELEMENTS "*")

foreach(ELEMENT ${ELEMENTS})
  if(IS_DIRECTORY ${ELEMENT} AND EXISTS ${ELEMENT}/CMakeLists.txt)

    file(TO_CMAKE_PATH ${ELEMENT} ELEMENT)
    get_filename_component(LAST_PATH ${ELEMENT} NAME)

    message(" ")
    message(STATUS "Start Building <${LAST_PATH}>.")
    configure_file(${LAST_PATH}/CMakeLists.txt ${LAST_PATH}/CMakeLists.txt)

    execute_process(
      COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -G Ninja .
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${LAST_PATH})
    execute_process(
      COMMAND ${CMAKE_COMMAND} --build .
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${LAST_PATH})
    message(STATUS "Finish Building <${LAST_PATH}>.")
  endif()
endforeach()

message(" ")
message(STATUS "Finish Building thirdparties.\n")
