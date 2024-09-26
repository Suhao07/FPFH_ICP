# cmake_minimum_required(VERSION 3.5)
# project(pcd_viewer)

# find_package(PCL 1.10 REQUIRED)

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# include_directories(${PCL_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
# link_directories(${PCL_LIBRARY_DIRS} ${Boost_LIBRARY_DIRS})
# add_definitions(${PCL_DEFINITIONS})

# ########## build the bin #############
# add_executable(viewer src/viewer.cpp)
# target_link_libraries(viewer ${PCL_LIBRARIES})

# add_executable(icp src/icp.cpp)
# target_link_libraries(icp ${PCL_LIBRARIES})

# add_executable(show_result src/show_result.cpp)
# target_link_libraries(show_result ${PCL_LIBRARIES})

# add_executable(FPFH src/FPFH.cpp)
# target_link_libraries(FPFH
#     ${PCL_LIBRARIES}
# )
# #######################################
cmake_minimum_required(VERSION 3.10)
project(PointCloudAlignment)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
find_package(PCL 1.10 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} include)

add_executable(point_cloud_alignment src/PointCloudAlignment.cpp src/main.cpp)
target_link_libraries(point_cloud_alignment ${PCL_LIBRARIES})



# 添加调试信息标志
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -g")
