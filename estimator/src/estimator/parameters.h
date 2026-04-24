/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of DRS-VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "../utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <map>

using namespace std;

const int WINDOW_SIZE = 4;
const int NUM_OF_F = 1000;
//#define UNIT_SPHERE_ERROR

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
extern int ESTIMATE_EXTRINSIC;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;
extern double FOCAL_LENGTH;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string OUTPUT_FOLDER;
extern std::string IMU_TOPIC;
extern double TD;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;
extern int ROW, COL;
extern int NUM_OF_CAM;
extern int STEREO;
extern int USE_IMU;
extern int MULTIPLE_THREAD;
// pts_gt for debug purpose;
extern map<int, Eigen::Vector3d> pts_gt;

extern std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern std::vector<double> CAM_FX;
extern std::vector<double> CAM_FY;
extern std::vector<double> CAM_CX;
extern std::vector<double> CAM_CY;
extern int MAX_CNT;
extern int MIN_DIST;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern int FLOW_BACK;
extern int FAST_THRESHOLD;

extern int DYNAMIC_WEIGHT_ENABLE;
extern double OPTICAL_FLOW_RESIDUAL_THRESHOLD;
extern double STEREO_RESIDUAL_THRESHOLD;
extern double OPTICAL_FLOW_NOISE_SIGMA;
extern double DISPARITY_NOISE_SIGMA;
extern double MIN_DYNAMIC_WEIGHT;
extern double DYNAMIC_WEIGHT_THRESHOLD;
extern double WEIGHT_DECAY_FACTOR;
extern double MIN_FUSED_WEIGHT;
extern double WEIGHT_SELECTION_THRESHOLD;
extern double CONFIDENCE_SIGMA_FACTOR;

extern int STRUCTURE_WEIGHT_ENABLE;
extern double STRUCTURE_WEIGHT_ALPHA;
extern double INFORMATION_ENTROPY_THRESHOLD;
extern double DEPTH_UNCERTAINTY_THRESHOLD;
extern double GEOMETRIC_STABILITY_THRESHOLD;
extern double PIXEL_NOISE_SIGMA;
extern double STRUCTURE_ENTROPY_EPSILON;
extern double MIN_STRUCTURE_WEIGHT;
extern double MAX_STRUCTURE_WEIGHT;

extern double OUTLIER_REJECTION_THRESHOLD;

struct FeatureObservation
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector3d point;
    Eigen::Vector2d uv;
    Eigen::Vector2d velocity;
    double optical_flow_residual;
    double stereo_residual;
    double dynamic_weight;
    double structure_weight;
    double depth_uncertainty;
    double information_entropy;
    double geometric_stability;

    FeatureObservation()
        : point(Eigen::Vector3d::Zero()),
          uv(Eigen::Vector2d::Zero()),
          velocity(Eigen::Vector2d::Zero()),
          optical_flow_residual(0.0),
          stereo_residual(0.0),
          dynamic_weight(1.0),
          structure_weight(1.0),
          depth_uncertainty(0.0),
          information_entropy(0.0),
          geometric_stability(0.0)
    {
    }
};

using FeatureMeasurement = std::pair<int, FeatureObservation>;
using FeatureFrame = std::map<int, std::vector<FeatureMeasurement>>;

void readParameters(std::string config_file);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
