cmake_minimum_required(VERSION 3.12)

project(show_ply)

find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
message(STATUS "${PCL_INCLUDE_DIRS}")
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (show_ply show_ply.cpp)
target_link_libraries (show_ply ${PCL_LIBRARIES})