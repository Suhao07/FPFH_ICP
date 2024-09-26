#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input.pcd>" << std::endl;
        return -1;
    }

    // 定义点云对象
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 读取PCD文件
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *cloud) == -1) {
        PCL_ERROR("Couldn't read file \n");
        return (-1);
    }

    std::cout << "Loaded " << cloud->width * cloud->height
              << " data points from " << argv[1] << std::endl;

    // 创建可视化对象并显示点云
    pcl::visualization::CloudViewer viewer("PCD Viewer");
    viewer.showCloud(cloud);

    while (!viewer.wasStopped()) {
        // 在窗口打开时保持循环
    }

    return 0;
}
