cmake_minimum_required(VERSION 3.10)
set(TARGET_NAME "OpenVINO-Detector")
project(${TARGET_NAME})
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(TBB_LIBS tbb)
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(TBB_LIBS tbb)
endif ()

if (NOT INSTALL_InferenceEngine_DIR)
    set(INSTALL_InferenceEngine_DIR "/usr/local")
endif ()

find_package(InferenceEngine REQUIRED PATHS ${INSTALL_InferenceEngine_DIR}/deployment_tools/inference_engine/share/)
include_directories(${InferenceEngine_INCLUDE_DIRS})
link_directories(${InferenceEngine_LIB_DIRS})
link_directories("${INSTALL_InferenceEngine_DIR}/deployment_tools/inference_engine/external/tbb/lib/")
find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})
set(INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/detection
        ${CMAKE_CURRENT_SOURCE_DIR}/ie
        )
file(GLOB SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/detection/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ie/*.cpp
        )

message("Sources: ${SOURCES}")
message("Include_dirs: ${INCLUDE_DIRS}")

include_directories(${INCLUDE_DIRS})
add_executable(${TARGET_NAME} ${SOURCES})
target_link_libraries(${TARGET_NAME} ${OpenCV_LIBRARIES} ${InferenceEngine_LIBRARIES} ${TBB_LIBS})