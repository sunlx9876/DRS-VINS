/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of DRS-VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#endif

namespace
{
double readNodeAsDouble(const cv::FileNode &node, const std::string &key, double fallback)
{
    const cv::FileNode child = node[key];
    return child.empty() ? fallback : double(child);
}

bool readCameraIntrinsics(const std::string &calib_path, double fallback_fx, double fallback_fy,
                          double fallback_cx, double fallback_cy,
                          double &fx, double &fy, double &cx, double &cy)
{
    fx = fallback_fx;
    fy = fallback_fy;
    cx = fallback_cx;
    cy = fallback_cy;

    cv::FileStorage fs(calib_path, cv::FileStorage::READ);
    if (!fs.isOpened())
        return false;

    const cv::FileNode projection = fs["projection_parameters"];
    if (projection.empty())
        return false;

    fx = readNodeAsDouble(projection, "fx", fallback_fx);
    fy = readNodeAsDouble(projection, "fy", fallback_fy);
    cx = readNodeAsDouble(projection, "cx", fallback_cx);
    cy = readNodeAsDouble(projection, "cy", fallback_cy);
    return true;
}

std::string expandUserPath(const std::string &path)
{
    if (path.empty() || path[0] != '~')
        return path;

    const char *home = std::getenv("HOME");
    if (!home)
        home = std::getenv("USERPROFILE");
    if (!home)
        return path;

    if (path.size() == 1)
        return std::string(home);
    if (path[1] == '/' || path[1] == '\\')
        return std::string(home) + path.substr(1);
    return path;
}

size_t findLastPathSeparator(const std::string &path)
{
    const size_t slash = path.find_last_of('/');
    const size_t backslash = path.find_last_of('\\');
    if (slash == std::string::npos)
        return backslash;
    if (backslash == std::string::npos)
        return slash;
    return std::max(slash, backslash);
}

std::string joinPath(const std::string &base, const std::string &child)
{
    if (child.empty())
        return base;
    if (child[0] == '/' || child[0] == '\\')
        return child;
    if (child.size() > 1 && child[1] == ':')
        return child;
    if (base.empty())
        return child;
    if (base.back() == '/' || base.back() == '\\')
        return base + child;
    return base + "/" + child;
}

bool directoryExists(const std::string &path)
{
    struct stat info;
    return stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR);
}

bool createSingleDirectory(const std::string &path)
{
    if (path.empty() || directoryExists(path))
        return true;
#ifdef _WIN32
    return _mkdir(path.c_str()) == 0 || errno == EEXIST;
#else
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
#endif
}

bool createDirectories(std::string path)
{
    path = expandUserPath(path);
    std::replace(path.begin(), path.end(), '\\', '/');
    while (path.size() > 1 && path.back() == '/')
        path.pop_back();
    if (path.empty())
        return true;

    std::string current;
    size_t index = 0;
    if (path.size() > 1 && path[1] == ':')
    {
        current = path.substr(0, 2);
        index = 2;
        if (index < path.size() && path[index] == '/')
        {
            current += "/";
            index++;
        }
    }
    else if (path[0] == '/')
    {
        current = "/";
        index = 1;
    }

    while (index < path.size())
    {
        const size_t next = path.find('/', index);
        const std::string part = path.substr(index, next == std::string::npos ? std::string::npos : next - index);
        if (!part.empty())
        {
            if (!current.empty() && current.back() != '/')
                current += "/";
            current += part;
            if (!createSingleDirectory(current))
                return false;
        }
        if (next == std::string::npos)
            break;
        index = next + 1;
    }
    return true;
}
} // namespace

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};
double FOCAL_LENGTH = 460.0;

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string IMU_TOPIC;
int ROW, COL;
double TD;
int NUM_OF_CAM;
int STEREO;
int USE_IMU;
int MULTIPLE_THREAD;
map<int, Eigen::Vector3d> pts_gt;
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
std::string FISHEYE_MASK;
std::vector<std::string> CAM_NAMES;
std::vector<double> CAM_FX;
std::vector<double> CAM_FY;
std::vector<double> CAM_CX;
std::vector<double> CAM_CY;
int MAX_CNT;
int MIN_DIST;
double F_THRESHOLD;
int SHOW_TRACK;
int FLOW_BACK;
int FAST_THRESHOLD;

int DYNAMIC_WEIGHT_ENABLE;
double OPTICAL_FLOW_RESIDUAL_THRESHOLD;
double STEREO_RESIDUAL_THRESHOLD;
double OPTICAL_FLOW_NOISE_SIGMA;
double DISPARITY_NOISE_SIGMA;
double MIN_DYNAMIC_WEIGHT;
double DYNAMIC_WEIGHT_THRESHOLD;
double WEIGHT_DECAY_FACTOR;
double MIN_FUSED_WEIGHT;
double WEIGHT_SELECTION_THRESHOLD;
double CONFIDENCE_SIGMA_FACTOR;

int STRUCTURE_WEIGHT_ENABLE;
double STRUCTURE_WEIGHT_ALPHA;
double INFORMATION_ENTROPY_THRESHOLD;
double DEPTH_UNCERTAINTY_THRESHOLD;
double GEOMETRIC_STABILITY_THRESHOLD;
double PIXEL_NOISE_SIGMA;
double STRUCTURE_ENTROPY_EPSILON;
double MIN_STRUCTURE_WEIGHT;
double MAX_STRUCTURE_WEIGHT;

double OUTLIER_REJECTION_THRESHOLD;


template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(std::string config_file)
{
    FILE *fh = fopen(config_file.c_str(),"r");
    if(fh == NULL){
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        ROS_BREAK();
        return;          
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["image0_topic"] >> IMAGE0_TOPIC;
    fsSettings["image1_topic"] >> IMAGE1_TOPIC;
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %d COL: %d ", ROW, COL);
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    FLOW_BACK = fsSettings["flow_back"];
    FAST_THRESHOLD = fsSettings["fast_threshold"].empty() ? 20 : int(fsSettings["fast_threshold"]);

    MULTIPLE_THREAD = fsSettings["multiple_thread"];

    USE_IMU = fsSettings["imu"];
    printf("USE_IMU: %d\n", USE_IMU);
    if(USE_IMU)
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;
        printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        G.z() = fsSettings["g_norm"];
    }

    SOLVER_TIME = fsSettings["max_solver_time"];
    NUM_ITERATIONS = fsSettings["max_num_iterations"];
    MIN_PARALLAX = fsSettings["keyframe_parallax"];

    fsSettings["output_path"] >> OUTPUT_FOLDER;
    OUTPUT_FOLDER = expandUserPath(OUTPUT_FOLDER);
    if (!createDirectories(OUTPUT_FOLDER))
        ROS_WARN_STREAM("Failed to create output directory: " << OUTPUT_FOLDER);
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    RIC.clear();
    TIC.clear();
    CAM_NAMES.clear();
    CAM_FX.clear();
    CAM_FY.clear();
    CAM_CX.clear();
    CAM_CY.clear();
    FOCAL_LENGTH = 460.0;

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    else 
    {
        if ( ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_T;
        fsSettings["body_T_cam0"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    } 
    
    NUM_OF_CAM = fsSettings["num_of_cam"];
    printf("camera number %d\n", NUM_OF_CAM);

    if(NUM_OF_CAM != 1 && NUM_OF_CAM != 2)
    {
        printf("num_of_cam should be 1 or 2\n");
        assert(0);
    }


    const size_t pn = findLastPathSeparator(config_file);
    const std::string configPath = pn == std::string::npos ? std::string(".") : config_file.substr(0, pn);
    
    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = joinPath(configPath, cam0Calib);
    CAM_NAMES.push_back(cam0Path);
    {
        double fx = 460.0;
        double fy = 460.0;
        double cx = COL / 2.0;
        double cy = ROW / 2.0;
        if (!readCameraIntrinsics(cam0Path, fx, fy, cx, cy, fx, fy, cx, cy))
            ROS_WARN_STREAM("Failed to read camera intrinsics from " << cam0Path << ", fallback to default intrinsics");
        CAM_FX.push_back(fx);
        CAM_FY.push_back(fy);
        CAM_CX.push_back(cx);
        CAM_CY.push_back(cy);
    }

    STEREO = NUM_OF_CAM == 2 ? 1 : 0;
    if(NUM_OF_CAM == 2)
    {
        std::string cam1Calib;
        fsSettings["cam1_calib"] >> cam1Calib;
        std::string cam1Path = joinPath(configPath, cam1Calib); 
        //printf("%s cam1 path\n", cam1Path.c_str() );
        CAM_NAMES.push_back(cam1Path);
        {
            double fx = CAM_FX.front();
            double fy = CAM_FY.front();
            double cx = CAM_CX.front();
            double cy = CAM_CY.front();
            if (!readCameraIntrinsics(cam1Path, fx, fy, cx, cy, fx, fy, cx, cy))
                ROS_WARN_STREAM("Failed to read camera intrinsics from " << cam1Path << ", fallback to left camera intrinsics");
            CAM_FX.push_back(fx);
            CAM_FY.push_back(fy);
            CAM_CX.push_back(cx);
            CAM_CY.push_back(cy);
        }
        
        cv::Mat cv_T;
        fsSettings["body_T_cam1"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }

    FOCAL_LENGTH = CAM_FX.empty() ? 460.0 : 0.5 * (CAM_FX.front() + CAM_FY.front());
    MIN_PARALLAX = MIN_PARALLAX / std::max(FOCAL_LENGTH, 1.0);

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    if(!USE_IMU)
    {
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    DYNAMIC_WEIGHT_ENABLE = fsSettings["dynamic_weight_enable"].empty() ? 1 : int(fsSettings["dynamic_weight_enable"]);
    OPTICAL_FLOW_RESIDUAL_THRESHOLD = fsSettings["optical_flow_residual_threshold"].empty()
                                          ? 0.5
                                          : double(fsSettings["optical_flow_residual_threshold"]);
    if (!fsSettings["optical_flow_threshold"].empty())
        OPTICAL_FLOW_RESIDUAL_THRESHOLD = double(fsSettings["optical_flow_threshold"]);
    STEREO_RESIDUAL_THRESHOLD = fsSettings["stereo_residual_threshold"].empty()
                                    ? 1.0
                                    : double(fsSettings["stereo_residual_threshold"]);
    if (!fsSettings["disparity_threshold"].empty())
        STEREO_RESIDUAL_THRESHOLD = double(fsSettings["disparity_threshold"]);
    OPTICAL_FLOW_NOISE_SIGMA = fsSettings["optical_flow_noise_sigma"].empty()
                                   ? OPTICAL_FLOW_RESIDUAL_THRESHOLD
                                   : double(fsSettings["optical_flow_noise_sigma"]);
    DISPARITY_NOISE_SIGMA = fsSettings["disparity_noise_sigma"].empty()
                                ? STEREO_RESIDUAL_THRESHOLD
                                : double(fsSettings["disparity_noise_sigma"]);
    MIN_DYNAMIC_WEIGHT = fsSettings["min_dynamic_weight"].empty() ? 0.1 : double(fsSettings["min_dynamic_weight"]);
    DYNAMIC_WEIGHT_THRESHOLD = fsSettings["dynamic_weight_threshold"].empty()
                                   ? 0.35
                                   : double(fsSettings["dynamic_weight_threshold"]);
    WEIGHT_DECAY_FACTOR = fsSettings["weight_decay_factor"].empty() ? 0.8 : double(fsSettings["weight_decay_factor"]);
    MIN_FUSED_WEIGHT = fsSettings["min_fused_weight"].empty() ? 1e-6 : double(fsSettings["min_fused_weight"]);
    WEIGHT_SELECTION_THRESHOLD = fsSettings["weight_selection_threshold"].empty()
                                     ? 0.2
                                     : double(fsSettings["weight_selection_threshold"]);
    CONFIDENCE_SIGMA_FACTOR = fsSettings["confidence_sigma_factor"].empty()
                                  ? 0.5
                                  : double(fsSettings["confidence_sigma_factor"]);

    STRUCTURE_WEIGHT_ENABLE = fsSettings["structure_weight_enable"].empty() ? 1 : int(fsSettings["structure_weight_enable"]);
    STRUCTURE_WEIGHT_ALPHA = fsSettings["structure_weight_alpha"].empty()
                                 ? 1.0
                                 : double(fsSettings["structure_weight_alpha"]);
    INFORMATION_ENTROPY_THRESHOLD = fsSettings["information_entropy_threshold"].empty()
                                        ? 0.5
                                        : double(fsSettings["information_entropy_threshold"]);
    DEPTH_UNCERTAINTY_THRESHOLD = fsSettings["depth_uncertainty_threshold"].empty()
                                      ? 0.1
                                      : double(fsSettings["depth_uncertainty_threshold"]);
    GEOMETRIC_STABILITY_THRESHOLD = fsSettings["geometric_stability_threshold"].empty()
                                        ? 0.5
                                        : double(fsSettings["geometric_stability_threshold"]);
    PIXEL_NOISE_SIGMA = fsSettings["pixel_noise_sigma"].empty()
                            ? 1.0
                            : double(fsSettings["pixel_noise_sigma"]);
    STRUCTURE_ENTROPY_EPSILON = fsSettings["structure_entropy_epsilon"].empty()
                                    ? 1e-6
                                    : double(fsSettings["structure_entropy_epsilon"]);
    MIN_STRUCTURE_WEIGHT = fsSettings["min_structure_weight"].empty()
                               ? 0.2
                               : double(fsSettings["min_structure_weight"]);
    MAX_STRUCTURE_WEIGHT = fsSettings["max_structure_weight"].empty()
                               ? 1.5
                               : double(fsSettings["max_structure_weight"]);

    OUTLIER_REJECTION_THRESHOLD = fsSettings["outlier_rejection_threshold"].empty()
                                      ? 3.0
                                      : double(fsSettings["outlier_rejection_threshold"]);

    fsSettings.release();
}
