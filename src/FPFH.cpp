#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <sstream>

class PointCloudRegistration {
public:
    PointCloudRegistration(const std::string& source_file, const std::string& target_file, const std::string& truth_file)
        : source_file_(source_file), target_file_(target_file), truth_file_(truth_file) {
        source_cloud_ = loadPointCloud(source_file_);
        target_cloud_ = loadPointCloud(target_file_);
    }

    // 运行配准流程
    void run() {
        source_normals_ = computeNormals(source_cloud_);
        target_normals_ = computeNormals(target_cloud_);

        source_fpfh_ = computeFPFHFeatures(source_cloud_, source_normals_);
        target_fpfh_ = computeFPFHFeatures(target_cloud_, target_normals_);

        estimated_transformation_ = estimateTransformationUsingFPFH(source_cloud_, source_fpfh_, target_cloud_, target_fpfh_);

        std::cout << "Estimated Transformation Matrix: \n" << estimated_transformation_ << std::endl;

        evaluateTransformation();

        transformed_cloud_ = transformPointCloud(source_cloud_, estimated_transformation_);

        visualizePointClouds();
    }

private:
    std::string source_file_;
    std::string target_file_;
    std::string truth_file_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_;
    pcl::PointCloud<pcl::Normal>::Ptr source_normals_;
    pcl::PointCloud<pcl::Normal>::Ptr target_normals_;
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr source_fpfh_;
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_fpfh_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud_;
    Eigen::Matrix4f estimated_transformation_;
public:
    // 读取真值矩阵
    Eigen::Matrix4f readTransformationMatrixFromFile(const std::string& filename) {
        Eigen::Matrix4f matrix;
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            throw std::runtime_error("无法打开文件");
        }
        std::string line;
        int i = 0;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#' || line.find("格式：PCD") != std::string::npos) {
                continue;
            }
            line.erase(0, line.find_first_not_of(' '));
            line.erase(line.find_last_not_of(' ') + 1);
            std::istringstream iss(line);
            float values[4];
            int j = 0;
            while (iss >> values[j] && j < 4) j++;
            if (j == 4) {
                for (int k = 0; k < 4; ++k) {
                    matrix(i, k) = values[k];
                }
                ++i;
            }
            if (i == 4) break;
        }
        file.close();
        return matrix;
    }

    // 计算矩阵差的Frobenius范数
    double computeMatrixDifference(const Eigen::Matrix4f &true_transformation, const Eigen::Matrix4f &estimated_transformation) {
        return (true_transformation - estimated_transformation).norm();
    }

    // 加载点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr loadPointCloud(const std::string& filename) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1) {
            std::cerr << "无法读取点云文件: " << filename << std::endl;
            throw std::runtime_error("无法读取点云文件");
        }
        return cloud;
    }

    // 计算法向量
    pcl::PointCloud<pcl::Normal>::Ptr computeNormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        normal_estimator.setSearchMethod(tree);
        normal_estimator.setInputCloud(cloud);
        normal_estimator.setKSearch(80);
        normal_estimator.compute(*normals);
        return normals;
    }

    // 计算FPFH特征
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr computeFPFHFeatures(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
        const pcl::PointCloud<pcl::Normal>::Ptr& normals) {
        pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs(new pcl::PointCloud<pcl::FPFHSignature33>);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        fpfh.setSearchMethod(tree);
        fpfh.setInputCloud(cloud);
        fpfh.setInputNormals(normals);
        fpfh.setKSearch(100);
        fpfh.compute(*fpfhs);
        return fpfhs;
    }

    // 使用FPFH和SVD估计变换矩阵
    Eigen::Matrix4f estimateTransformationUsingFPFH(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud, 
        const pcl::PointCloud<pcl::FPFHSignature33>::Ptr& source_fpfh, 
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud, 
        const pcl::PointCloud<pcl::FPFHSignature33>::Ptr& target_fpfh) {
        
        pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());
        pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> corr_est;
        corr_est.setInputSource(source_fpfh);
        corr_est.setInputTarget(target_fpfh);
        corr_est.determineReciprocalCorrespondences(*correspondences);

        pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> trans_est;
        Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
        trans_est.estimateRigidTransformation(*source_cloud, *target_cloud, *correspondences, transformation);
        return transformation;
    }

    // 变换点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const Eigen::Matrix4f& transformation) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*cloud, *transformed_cloud, transformation);
        return transformed_cloud;
    }

    // 可视化点云
    void visualizePointClouds() {
        pcl::visualization::PCLVisualizer viewer("FPFH + Transformation Visualization");

        viewer.setBackgroundColor(0.0, 0.0, 0.0);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(source_cloud_, 255, 0, 0); // 红色
        viewer.addPointCloud(source_cloud_, source_color, "source_cloud");
        viewer.addText("Source Cloud", 10, 570, 25, 1, 0, 0, "source_text");

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(target_cloud_, 0, 0, 255); // 蓝色
        viewer.addPointCloud(target_cloud_, target_color, "target_cloud");
        viewer.addText("Target Cloud", 10, 540, 25, 0, 0, 1, "target_text");

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transformed_color(transformed_cloud_, 0, 255, 0); // 绿色
        viewer.addPointCloud(transformed_cloud_, transformed_color, "transformed_cloud");
        viewer.addText("Transformed Cloud", 10, 510, 25, 0, 1, 0, "transformed_text");

        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source_cloud");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target_cloud");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "transformed_cloud");

        while (!viewer.wasStopped()) {
            viewer.spinOnce();
        }
    }

    // 评估估计的变换矩阵
    void evaluateTransformation() {
        Eigen::Matrix4f true_transformation = readTransformationMatrixFromFile(truth_file_);
        double frobenius_norm = computeMatrixDifference(true_transformation, estimated_transformation_);
        std::cout << "Frobenius norm of transformation difference: " << frobenius_norm << std::endl;
    }
};

// 主函数
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "用法: " << argv[0] << " <source_cloud.pcd> <target_cloud.pcd> <truth.txt>" << std::endl;
        return -1;
    }

    std::string source_file = argv[1];
    std::string target_file = argv[2];
    std::string truth_file = argv[3];

    PointCloudRegistration registration(source_file, target_file, truth_file);
    registration.run();

    return 0;
}
