#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>  
#include "PointCloudAlignment.hpp"

int main(int argc, char** argv) {
    // 检查输入参数
    if (argc != 3) {
        std::cerr << "用法: " << argv[0] << " <源点云文件> <目标点云文件>" << std::endl;
        return -1;
    }

    
    // 加载输入点云 (视为相机下的物体深度点云)
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>());  // 目标点云

    // 加载源点云 (PLY 格式)
    if (pcl::io::loadPLYFile<pcl::PointXYZ>(argv[1], *input_cloud) == -1) {
        PCL_ERROR("无法加载输入点云文件！\n");
        return -1;
    }

    // 加载目标点云 (标准姿态下的物体模型，PLY 格式)
    if (pcl::io::loadPLYFile<pcl::PointXYZ>(argv[2], *target_cloud) == -1) {
        PCL_ERROR("无法加载目标点云文件！\n");
        return -1;
    }

    // 创建PointCloudAlignment对象
    PointCloudAlignment aligner(0.1, 100, 0.02); // 自定义超参数

    // 执行点云对齐
    Eigen::Matrix4f transformation = aligner.alignPointClouds(input_cloud, target_cloud);

    std::cout << "最终变换矩阵：\n" << transformation << std::endl;

    return 0;
}
// TO DO::
// 修改代码，估计输入点云到相机的相对位姿
// 封装成ros节点
//如果看到的是局部特征应该怎么处理呢？比如局部特征的点云配准
