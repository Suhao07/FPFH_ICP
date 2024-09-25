#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Dense>
#include "PointCloudAlignment.hpp"

// 构造函数
PointCloudAlignment::PointCloudAlignment(float search_radius, int max_iterations, float max_correspondence_distance) 
    : search_radius(search_radius), max_iterations(max_iterations), max_correspondence_distance(max_correspondence_distance) {}

// 提取FPFH特征
pcl::PointCloud<pcl::FPFHSignature33>::Ptr PointCloudAlignment::extractFPFH(
                                        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                        pcl::PointCloud<pcl::Normal>::Ptr normals) {
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(cloud);
    fpfh.setInputNormals(normals);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    fpfh.setSearchMethod(tree);
    
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_features(new pcl::PointCloud<pcl::FPFHSignature33>);
    fpfh.setRadiusSearch(search_radius); // 使用可调节的搜索半径
    fpfh.compute(*fpfh_features);
    
    return fpfh_features;
}

// 粗配准：使用FPFH特征和SAC-IA
Eigen::Matrix4f PointCloudAlignment::coarseRegistration(pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud,
                                       pcl::PointCloud<pcl::FPFHSignature33>::Ptr source_features,
                                       pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud,
                                       pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_features) {
    pcl::SampleConsensusPrerejective<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia;
    sac_ia.setInputSource(source_cloud);
    sac_ia.setSourceFeatures(source_features);
    sac_ia.setInputTarget(target_cloud);
    sac_ia.setTargetFeatures(target_features);

    sac_ia.setMaximumIterations(max_iterations); // 使用可调节的最大迭代次数
    sac_ia.setNumberOfSamples(3);               
    sac_ia.setCorrespondenceRandomness(5);
    sac_ia.setSimilarityThreshold(0.9f);
    sac_ia.setMaxCorrespondenceDistance(max_correspondence_distance); // 使用可调节的最大对应距离
    sac_ia.setInlierFraction(0.25f);

    pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
    sac_ia.align(aligned_cloud);

    if (sac_ia.hasConverged()) {
        std::cout << "粗配准成功!" << std::endl;
        return sac_ia.getFinalTransformation();
    } else {
        std::cerr << "粗配准失败" << std::endl;
        return Eigen::Matrix4f::Identity();
    }
}

// 精细配准：基于ICP
Eigen::Matrix4f PointCloudAlignment::fineRegistration(pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud,
                                                    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud,
                                                    Eigen::Matrix4f coarse_transformation) {
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    pcl::PointCloud<pcl::PointXYZ> aligned_cloud;

    icp.setInputSource(source_cloud);
    icp.setInputTarget(target_cloud);
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-8);
    icp.setEuclideanFitnessEpsilon(1e-8);
    icp.align(aligned_cloud, coarse_transformation);

    if (icp.hasConverged()) {
        std::cout << "精细配准成功!" << std::endl;
        return icp.getFinalTransformation();
    } else {
        std::cerr << "精细配准失败" << std::endl;
        return Eigen::Matrix4f::Identity();
    }
}

// 新增函数：在相机点云中匹配物体点云
pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudAlignment::matchObject(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud,
                                                pcl::PointCloud<pcl::FPFHSignature33>::Ptr input_features,
                                                pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_features) {
    // 使用一些特征匹配算法找到与目标物体相关的点云
    pcl::SampleConsensusPrerejective<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia;
    sac_ia.setInputSource(input_cloud);
    sac_ia.setSourceFeatures(input_features);
    sac_ia.setTargetFeatures(target_features);

    sac_ia.setMaximumIterations(max_iterations); // 使用可调节的最大迭代次数
    sac_ia.setMaxCorrespondenceDistance(max_correspondence_distance); // 可调节的最大对应距离
    sac_ia.setNumberOfSamples(3);               
    sac_ia.setCorrespondenceRandomness(5);
    sac_ia.setSimilarityThreshold(0.9f);
    sac_ia.setInlierFraction(0.25f);

    pcl::PointCloud<pcl::PointXYZ>::Ptr matched_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    sac_ia.align(*matched_cloud);

    if (sac_ia.hasConverged()) {
        std::cout << "物体匹配成功！" << std::endl;
        return matched_cloud;
    } else {
        std::cerr << "物体匹配失败" << std::endl;
        return nullptr;
    }
}


// 修改：计算物体相对于相机的相对位姿
Eigen::Matrix4f PointCloudAlignment::alignPointClouds(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud,
                                                        pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud) {
    // 1. 计算法线
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_est;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::PointCloud<pcl::Normal>::Ptr input_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Normal>::Ptr target_normals(new pcl::PointCloud<pcl::Normal>);

    normal_est.setInputCloud(input_cloud);
    normal_est.setSearchMethod(tree);
    normal_est.setRadiusSearch(0.03);  // 设置法线估计的搜索半径
    normal_est.compute(*input_normals);

    normal_est.setInputCloud(target_cloud);
    normal_est.compute(*target_normals);

    // 2. 提取FPFH特征
    auto input_features = extractFPFH(input_cloud, input_normals);
    auto target_features = extractFPFH(target_cloud, target_normals);

    // 3. 匹配物体并提取匹配的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr matched_cloud = matchObject(input_cloud, input_features, target_features);

    if (!matched_cloud || matched_cloud->points.empty()) {
        std::cerr << "未找到匹配的物体点云" << std::endl;
        return Eigen::Matrix4f::Identity();
    }

    // 4. 粗配准
    Eigen::Matrix4f coarse_transformation = coarseRegistration(matched_cloud, input_features, target_cloud, target_features);

    // 5. 精细配准
    Eigen::Matrix4f fine_transformation = fineRegistration(matched_cloud, target_cloud, coarse_transformation);

    // 返回从物体到相机的相对位姿
    return fine_transformation;
}