cmake_minimum_required(VERSION 2.8)
project(adaptive_gftt)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(OpenCV 3 REQUIRED)
message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")
include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})

add_executable(adaptive_gftt adaptive_gftt.cpp gftt.cpp)
target_link_libraries(adaptive_gftt ${OpenCV_LIBS})
