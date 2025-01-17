#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <Eigen/Dense>
#include <pcl/common/common.h>  

#include "PointCloudAlignment.hpp"

// 构造函数
PointCloudAlignment::PointCloudAlignment(float search_radius, int max_iterations, float max_correspondence_distance) 
    : search_radius(search_radius), max_iterations(max_iterations), max_correspondence_distance(max_correspondence_distance) {}

// 点云预处理：下采样和去噪
pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudAlignment::preprocessPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    // 1. 下采样
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    // voxel_grid.setLeafSize(0.01f, 0.01f, 0.01f);  // 可调节的体素大小
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(*cloud, min_pt, max_pt);
    float leaf_size = (max_pt.x - min_pt.x) * 0.01; // 动态调整 leaf size，取点云范围的1%
    voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    voxel_grid.filter(*filtered_cloud);
    
    // 2. 去噪
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(filtered_cloud);
    sor.setMeanK(50);  // 平均距离计算中使用的邻居数量
    sor.setStddevMulThresh(1.0);  // 离群点判断阈值
    sor.filter(*filtered_cloud);
    
    return filtered_cloud;
}

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
// Eigen::Matrix4f PointCloudAlignment::coarseRegistration(pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud,
//                                        pcl::PointCloud<pcl::FPFHSignature33>::Ptr source_features,
//                                        pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud,
//                                        pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_features) {
//     pcl::SampleConsensusPrerejective<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia;
//     sac_ia.setInputSource(source_cloud);
//     sac_ia.setSourceFeatures(source_features);
//     sac_ia.setInputTarget(target_cloud);
//     sac_ia.setTargetFeatures(target_features);

//     sac_ia.setMaximumIterations(max_iterations); // 使用可调节的最大迭代次数
//     sac_ia.setNumberOfSamples(3);               
//     sac_ia.setCorrespondenceRandomness(5);
//     sac_ia.setSimilarityThreshold(0.9f);
//     sac_ia.setMaxCorrespondenceDistance(max_correspondence_distance); // 使用可调节的最大对应距离
//     sac_ia.setInlierFraction(0.25f);

//     pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
//     sac_ia.align(aligned_cloud);

//     if (sac_ia.hasConverged()) {
//         std::cout << "粗配准成功!" << std::endl;
//         return sac_ia.getFinalTransformation();
//     } else {
//         std::cerr << "粗配准失败" << std::endl;
//         return Eigen::Matrix4f::Identity();
//     }
// }
Eigen::Matrix4f PointCloudAlignment::coarseRegistration(
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud,
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr source_features,
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud,
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_features) {
    
    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());

    // const int max_iterations = 50;
    const float convergence_threshold = 1e-6;

    for (int i = 0; i < max_iterations; ++i) {
        // 根据当前变换更新源点云
        pcl::transformPointCloud(*source_cloud, *transformed_cloud, transformation);

        // 计算对应关系
        correspondences->clear(); // 清空之前的对应关系
        pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> corr_est;
        corr_est.setInputSource(source_features);
        corr_est.setInputTarget(target_features);
        corr_est.determineReciprocalCorrespondences(*correspondences);

        // 使用 SVD 估计新的变换矩阵
        pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> trans_est;
        Eigen::Matrix4f new_transformation;
        trans_est.estimateRigidTransformation(*transformed_cloud, *target_cloud, *correspondences, new_transformation);

        // 更新总变换
        transformation = new_transformation * transformation;

        // 检查收敛条件
        if ((new_transformation.block<3, 1>(0, 3)).norm() < convergence_threshold) {
            std::cout << "粗配准成功，迭代次数: " << i + 1 << std::endl;
            return transformation;
        }
    }

    std::cerr << "粗配准失败，达到最大迭代次数" << std::endl;
    return Eigen::Matrix4f::Identity();
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

// 计算物体相对于相机的相对位姿
Eigen::Matrix4f PointCloudAlignment::alignPointClouds(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud,
                                                        pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud,
                                                        const std::string& output_filename) {
    // 1. 对输入点云和目标点云进行预处理
    input_cloud = preprocessPointCloud(input_cloud);
    target_cloud = preprocessPointCloud(target_cloud);

    // 2. 计算法线
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

    // 3. 提取FPFH特征
    auto input_features = extractFPFH(input_cloud, input_normals);
    auto target_features = extractFPFH(target_cloud, target_normals);

    // 4. 粗配准
    Eigen::Matrix4f coarse_transformation = coarseRegistration(input_cloud, input_features, target_cloud, target_features);

    // 5. 精细配准
    Eigen::Matrix4f fine_transformation = fineRegistration(input_cloud, target_cloud, coarse_transformation);
     // Fine registration
    transformation_matrix = fine_transformation; // Store transformation matrix
      // 6. 保存配准后的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*input_cloud, *transformed_cloud, fine_transformation);

    // Save the aligned point cloud to a PLY file
    if (pcl::io::savePLYFile(output_filename, *transformed_cloud) == 0) {
        std::cout << "保存配准后的点云成功: " << output_filename << std::endl;
    } else {
        std::cerr << "保存配准后的点云失败." << std::endl;
    }

    // 返回从物体到相机的相对位姿
    return fine_transformation;
}

void PointCloudAlignment::visualizePointClouds(pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud,
                                                pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud) {
    pcl::visualization::PCLVisualizer viewer("Point Cloud Alignment");

   
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(source_cloud, 255, 0, 0); // Red
    viewer.addPointCloud(source_cloud, source_color, "source_cloud");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(target_cloud, 0, 255, 0); // Green
    viewer.addPointCloud(target_cloud, target_color, "target_cloud");

   
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    
   
    pcl::transformPointCloud(*source_cloud, *transformed_cloud, transformation_matrix);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> aligned_color(transformed_cloud, 0, 0, 255); // Blue
    viewer.addPointCloud(transformed_cloud, aligned_color, "aligned_cloud");

   
    viewer.addCoordinateSystem(1.0);
    viewer.setBackgroundColor(1.0, 1.0, 1.0); 
    viewer.initCameraParameters(); 

    
    while (!viewer.wasStopped()) {
        viewer.spinOnce(100); 
    }
}
