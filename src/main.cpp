#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <iostream>
#include "PointCloudAlignment.hpp"



int main(int argc, char** argv) {
    // 检查输入参数
    if (argc != 4) { // Updated to expect three arguments
        std::cerr << "用法: " << argv[0] << " <源点云文件> <目标点云文件> <输出点云文件>" << std::endl;
        return -1;
    }

    // 创建点云对象
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 加载源点云
    if (pcl::io::loadPLYFile(argv[1], *source_cloud) == -1) {
        PCL_ERROR("无法加载源点云文件\n");
        return -1;
    }

    // 加载目标点云
    if (pcl::io::loadPLYFile(argv[2], *target_cloud) == -1) {
        PCL_ERROR("无法加载目标点云文件\n");
        return -1;
    }

    // 创建PointCloudAlignment对象
    float search_radius = 0.02f; // 根据需要设置
    int max_iterations = 1000; // 根据需要设置
    float max_correspondence_distance = 0.05f; // 根据需要设置

    PointCloudAlignment pointCloudAligner(search_radius, max_iterations, max_correspondence_distance);

    // 执行点云配准并保存结果
    Eigen::Matrix4f transformation = pointCloudAligner.alignPointClouds(source_cloud, target_cloud, argv[3]);

    std::cout << "最终变换矩阵: \n" << transformation << std::endl;

    return 0;
}

// TO DO::
// 修改代码，估计输入点云到相机的相对位姿
// 封装成ros节点
//如果看到的是局部特征应该怎么处理呢？比如局部特征的点云配准
