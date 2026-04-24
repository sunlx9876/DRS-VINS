/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of DRS-VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"
#include "../utility/tic_toc.h"

class FeaturePerFrame
{
  public:
    FeaturePerFrame(const FeatureObservation &_observation, double td)
    {
        point = _observation.point;
        uv = _observation.uv;
        velocity = _observation.velocity;
        optical_flow_residual = _observation.optical_flow_residual;
        stereo_residual = _observation.stereo_residual;
        dynamic_weight = _observation.dynamic_weight;
        cur_td = td;
        is_stereo = false;
    }
    void rightObservation(const FeatureObservation &_observation)
    {
        pointRight = _observation.point;
        uvRight = _observation.uv;
        velocityRight = _observation.velocity;
        stereo_residual = _observation.stereo_residual;
        dynamic_weight = std::min(dynamic_weight, _observation.dynamic_weight);
        is_stereo = true;
    }
    double cur_td;
    Vector3d point, pointRight;
    Vector2d uv, uvRight;
    Vector2d velocity, velocityRight;
    double optical_flow_residual;
    double stereo_residual;
    double dynamic_weight;
    bool is_stereo;
};

class FeaturePerId
{
  public:
    const int feature_id;
    int start_frame;
    vector<FeaturePerFrame> feature_per_frame;
    int used_num;
    double estimated_depth;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;
    double structure_weight;
    double depth_uncertainty;
    double information_entropy;
    double geometric_stability;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0),
          structure_weight(1.0), depth_uncertainty(0.0),
          information_entropy(0.0), geometric_stability(0.0)
    {
    }

    int endFrame();
};

class FeatureManager
{
  public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);
    void clearState();
    int getFeatureCount();
    bool addFeatureCheckParallax(int frame_count, const FeatureFrame &image, double td);
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);
    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);
    void removeFailures();
    void clearDepth();
    VectorXd getDepthVector();
    void triangulate(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[]);
    void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                            Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d);
    void initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[]);
    bool solvePoseByPnP(Eigen::Matrix3d &R_initial, Eigen::Vector3d &P_initial, 
                            vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier(set<int> &outlierIndex);
    list<FeaturePerId> feature;
    int last_track_num;
    double last_average_parallax;
    int new_feature_num;
    int long_track_num;

  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    double computeDepthUncertainty(double baseline, double disparity) const;
    Eigen::Matrix3d computePointCovariance(const Eigen::Vector2d &left_uv, const Eigen::Vector2d &right_uv, double baseline) const;
    double computeInformationEntropy(const Eigen::Matrix3d &point_covariance) const;
    double computeStructureWeight(double information_entropy) const;
    const Matrix3d *Rs;
    Matrix3d ric[2];
};

#endif
