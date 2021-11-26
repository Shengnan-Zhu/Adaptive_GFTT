cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(OpenCV 3 REQUIRED)
message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")
include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})

add_executable(adaptive_gftt adaptive_gftt.cpp)