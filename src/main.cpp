#include <pcl/io/pcd_io.h>
#include "PointCloudAlignment.hpp"

int main(int argc, char** argv) {
    // 检查输入参数
    if (argc != 3) {
        std::cerr << "用法: " << argv[0] << " <源点云文件> <目标点云文件>" << std::endl;
        return -1;
    }

    // 加载点云数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>());

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *source_cloud) == -1) {
        PCL_ERROR("无法加载源点云文件！\n");
        return -1;
    }

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[2], *target_cloud) == -1) {
        PCL_ERROR("无法加载目标点云文件！\n");
        return -1;
    }

    // 创建PointCloudAlignment对象
    PointCloudAlignment aligner(0.1, 100000, 0.02); // 自定义超参数

    // 执行点云对齐
    Eigen::Matrix4f transformation = aligner.alignPointClouds(source_cloud, target_cloud);

    std::cout << "最终变换矩阵：\n" << transformation << std::endl;

    return 0;
}
