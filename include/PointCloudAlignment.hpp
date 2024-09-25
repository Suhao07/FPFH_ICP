#ifndef POINTCLOUDALIGNMENT_HPP
#define POINTCLOUDALIGNMENT_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <Eigen/Dense>

class PointCloudAlignment {
public:
   
    PointCloudAlignment(float search_radius = 0.05, int max_iterations = 50000, float max_correspondence_distance = 0.01);
    
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr extractFPFH(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals);
    
    // 粗配准：使用FPFH特征和SAC-IA
    Eigen::Matrix4f coarseRegistration(pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud,
                                       pcl::PointCloud<pcl::FPFHSignature33>::Ptr source_features,
                                       pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud,
                                       pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_features);
    
    // 精细配准：基于ICP
    Eigen::Matrix4f fineRegistration(pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud,
                                     pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud,
                                     Eigen::Matrix4f coarse_transformation);
    
    // 计算物体相对于相机的相对位姿
    Eigen::Matrix4f alignPointClouds(pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud,
                                     pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud);
    
   
    float search_radius;
    int max_iterations;
    float max_correspondence_distance;
};

#endif
