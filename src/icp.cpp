#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Dense>

class PointCloudRegistration {
public:
    // 读取真值矩阵
    static Eigen::Matrix4f readTransformationMatrixFromFile(const std::string& filename) {
        Eigen::Matrix4f matrix;
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            throw std::runtime_error("无法打开文件");
        }
        std::string line;
        int i = 0;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            if (line.find("格式：PCD") != std::string::npos || 
                line.find("点数") != std::string::npos || 
                line.find("由源点云变换到目标点云的旋转平移矩阵") != std::string::npos || 
                line.find("在X轴方向上旋转了45°") != std::string::npos) {
                continue;
            }
            line.erase(0, line.find_first_not_of(' '));
            line.erase(line.find_last_not_of(' ') + 1);
            std::istringstream iss(line);
            float values[4];
            int j = 0;
            while (iss >> values[j] && j < 4) {
                j++;
            }
            if (j == 4) {
                for (int k = 0; k < 4; ++k) {
                    matrix(i, k) = values[k];
                }
                ++i;
            }
            if (i == 4) break;
        }
        if (i != 4) {
            std::cerr << "文件格式错误: " << filename << " - Expected 4 lines but got " << i << std::endl;
            throw std::runtime_error("文件格式错误");
        }
        file.close();
        return matrix;
    }

    // 计算矩阵差异的Frobenius范数
    static double computeMatrixDifference(const Eigen::Matrix4f &true_transformation, const Eigen::Matrix4f &estimated_transformation) {
        return (true_transformation - estimated_transformation).norm();
    }

    // 执行ICP配准并输出结果
    void performICP(const std::string& source_file, const std::string& target_file, const std::string& truth_file) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);

        if (pcl::io::loadPCDFile<pcl::PointXYZ>(source_file, *source_cloud) == -1 ||
            pcl::io::loadPCDFile<pcl::PointXYZ>(target_file, *target_cloud) == -1) {
            std::cerr << "无法读取源或目标点云文件。" << std::endl;
            return;
        }

        // 初始化ICP对象
        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setInputSource(source_cloud);
        icp.setInputTarget(target_cloud);

        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-8);
        icp.setEuclideanFitnessEpsilon(1e-8);

        pcl::PointCloud<pcl::PointXYZ> final_cloud;
        icp.align(final_cloud);

        // 结果变换矩阵
        Eigen::Matrix4f estimated_matrix = icp.getFinalTransformation();
        std::cout << "ICP 配准变换矩阵:" << std::endl << estimated_matrix << std::endl;

        // 读取真值矩阵
        Eigen::Matrix4f truth_matrix = readTransformationMatrixFromFile(truth_file);
        double error = computeMatrixDifference(truth_matrix, estimated_matrix);
        std::cout << "矩阵差异的Frobenius范数: " << error << std::endl;

        // 计算均方根误差
        double fitness_score = icp.getFitnessScore();
        std::cout << "均方根误差 (RMSE): " << fitness_score << std::endl;

        visualizePointClouds(source_cloud, target_cloud, final_cloud);
    }

    // 可视化点云
    void visualizePointClouds(pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud,
                              pcl::PointCloud<pcl::PointXYZ>& final_cloud) {
        pcl::visualization::PCLVisualizer viewer("ICP Visualization");

        pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>(final_cloud));

        // 可视化源点云
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(source_cloud, 255, 0, 0); // 红色
        viewer.addPointCloud(source_cloud, source_color, "source_cloud");
        viewer.addText("Source Cloud", 10, 570, 25, 1, 0, 0, "source_text");

        // 可视化目标点云
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(target_cloud, 0, 0, 255); // 蓝色
        viewer.addPointCloud(target_cloud, target_color, "target_cloud");
        viewer.addText("Target Cloud", 10, 540, 25, 0, 0, 1, "target_text");

        // 可视化配准后的点云
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> final_color(result, 0, 255, 0); // 绿色
        viewer.addPointCloud(result, final_color, "Final");
        viewer.addText("Transformed Cloud", 10, 510, 25, 0, 1, 0, "transformed_text");

        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "source_cloud");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "target_cloud");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "Final");

        viewer.setSize(800, 600);

        while (!viewer.wasStopped()) {
            viewer.spinOnce();
        }
    }
};

// 主函数
int main(int argc, char** argv) {
    if (argc != 6) {
        std::cout << "Usage: " << argv[0] << " <source_cloud.pcd> <target_cloud.pcd> <output_cloud.pcd> <output_matrix.txt> <truth.txt>" << std::endl;
        return -1;
    }

    PointCloudRegistration icp_registration;
    icp_registration.performICP(argv[1], argv[2], argv[5]);

    return 0;
}
