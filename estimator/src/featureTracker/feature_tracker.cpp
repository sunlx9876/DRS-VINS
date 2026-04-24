/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of DRS-VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "feature_tracker.h"

#include <cmath>

bool FeatureTracker::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

double distance(cv::Point2f pt1, cv::Point2f pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<double> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker()
{
    stereo_cam = 0;
    n_id = 0;
    hasPrediction = false;
}

void FeatureTracker::setMask()
{
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
    struct TrackSlot
    {
        int track_count;
        cv::Point2f point;
        int id;
        double flow_residual;
        double stereo_residual;
        double dynamic_weight;
    };
    vector<TrackSlot> cnt_pts_id;

    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        TrackSlot slot;
        slot.track_count = track_cnt[i];
        slot.point = cur_pts[i];
        slot.id = ids[i];
        slot.flow_residual = i < optical_flow_residuals.size() ? optical_flow_residuals[i] : 0.0;
        slot.stereo_residual = i < stereo_residuals.size() ? stereo_residuals[i] : 0.0;
        slot.dynamic_weight = i < dynamic_weights.size() ? dynamic_weights[i] : 1.0;
        cnt_pts_id.push_back(slot);
    }

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const TrackSlot &a, const TrackSlot &b)
         {
            return a.track_count > b.track_count;
         });

    cur_pts.clear();
    ids.clear();
    track_cnt.clear();
    optical_flow_residuals.clear();
    stereo_residuals.clear();
    dynamic_weights.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.point) == 255)
        {
            cur_pts.push_back(it.point);
            ids.push_back(it.id);
            track_cnt.push_back(it.track_count);
            optical_flow_residuals.push_back(it.flow_residual);
            stereo_residuals.push_back(it.stereo_residual);
            dynamic_weights.push_back(it.dynamic_weight);
            cv::circle(mask, it.point, MIN_DIST, 0, -1);
        }
    }
}

double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

FeatureFrame FeatureTracker::trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1)
{
    TicToc t_r;
    cur_time = _cur_time;
    cur_img = _img;
    row = cur_img.rows;
    col = cur_img.cols;
    cv::Mat rightImg = _img1;
    /*
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(cur_img, cur_img);
        if(!rightImg.empty())
            clahe->apply(rightImg, rightImg);
    }
    */
    cur_pts.clear();
    optical_flow_residuals.clear();
    stereo_residuals.clear();
    dynamic_weights.clear();

    if (prev_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        vector<double> flow_residuals(prev_pts.size(), 0.0);
        if(hasPrediction)
        {
            cur_pts = predict_pts;
            cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 1, 
            cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
            
            int succ_num = 0;
            for (size_t i = 0; i < status.size(); i++)
            {
                if (status[i])
                    succ_num++;
            }
            if (succ_num < 10)
               cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
        }
        else
            cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
        // reverse check
        if(FLOW_BACK)
        {
            vector<uchar> reverse_status;
            vector<cv::Point2f> reverse_pts = prev_pts;
            cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 1, 
            cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
            //cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 3); 
            for(size_t i = 0; i < status.size(); i++)
            {
                double reverse_error = reverse_status[i] ? distance(prev_pts[i], reverse_pts[i]) : 2.0 * OPTICAL_FLOW_RESIDUAL_THRESHOLD;
                flow_residuals[i] = reverse_error;
                if(status[i] && reverse_status[i] && reverse_error <= OPTICAL_FLOW_RESIDUAL_THRESHOLD)
                {
                    status[i] = 1;
                }
                else
                    status[i] = 0;
            }
        }
        else
        {
            for (size_t i = 0; i < status.size(); i++)
            {
                flow_residuals[i] = i < err.size() ? err[i] : 0.0;
            }
        }
        
        for (int i = 0; i < int(cur_pts.size()); i++)
            if (status[i] && !inBorder(cur_pts[i]))
                status[i] = 0;
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        reduceVector(flow_residuals, status);
        optical_flow_residuals = flow_residuals;
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
        //printf("track cnt %d\n", (int)ids.size());
    }

    for (auto &n : track_cnt)
        n++;

    if (optical_flow_residuals.size() != cur_pts.size())
        optical_flow_residuals.assign(cur_pts.size(), 0.0);
    stereo_residuals.assign(cur_pts.size(), 0.0);
    dynamic_weights.assign(cur_pts.size(), 1.0);

    if (1)
    {
        //rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            vector<cv::KeyPoint> keypoints;
            cv::FAST(cur_img, keypoints, FAST_THRESHOLD, true);
            sort(keypoints.begin(), keypoints.end(), [](const cv::KeyPoint &lhs, const cv::KeyPoint &rhs)
                 {
                     return lhs.response > rhs.response;
                 });

            cv::Mat detection_mask = mask.clone();
            n_pts.clear();
            for (const auto &keypoint : keypoints)
            {
                cv::Point pixel(cvRound(keypoint.pt.x), cvRound(keypoint.pt.y));
                if (!inBorder(keypoint.pt))
                    continue;
                if (detection_mask.at<uchar>(pixel.y, pixel.x) == 0)
                    continue;

                n_pts.push_back(keypoint.pt);
                cv::circle(detection_mask, keypoint.pt, MIN_DIST, 0, -1);
                if (static_cast<int>(n_pts.size()) >= n_max_cnt)
                    break;
            }
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %f ms", t_t.toc());

        for (auto &p : n_pts)
        {
            cur_pts.push_back(p);
            ids.push_back(n_id++);
            track_cnt.push_back(1);
            optical_flow_residuals.push_back(0.0);
            stereo_residuals.push_back(0.0);
            dynamic_weights.push_back(1.0);
        }
        //printf("feature cnt after add %d\n", (int)ids.size());
    }

    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    if(!_img1.empty() && stereo_cam)
    {
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();
        if(!cur_pts.empty())
        {
            //printf("stereo image; track feature on right image\n");
            vector<cv::Point2f> reverseLeftPts;
            vector<uchar> status, statusRightLeft;
            vector<float> err;
            vector<double> right_residuals(cur_pts.size(), 2.0 * STEREO_RESIDUAL_THRESHOLD);
            // cur left ---- cur right
            cv::calcOpticalFlowPyrLK(cur_img, rightImg, cur_pts, cur_right_pts, status, err, cv::Size(21, 21), 3);
            // reverse check cur right ---- cur left
            if(FLOW_BACK)
            {
                cv::calcOpticalFlowPyrLK(rightImg, cur_img, cur_right_pts, reverseLeftPts, statusRightLeft, err, cv::Size(21, 21), 3);
                for(size_t i = 0; i < status.size(); i++)
                {
                    double reverse_error = statusRightLeft[i] ? distance(cur_pts[i], reverseLeftPts[i]) : 2.0 * STEREO_RESIDUAL_THRESHOLD;
                    double y_error = status[i] ? std::abs(cur_pts[i].y - cur_right_pts[i].y) : 2.0 * STEREO_RESIDUAL_THRESHOLD;
                    right_residuals[i] = y_error + reverse_error;
                    if(status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && reverse_error <= STEREO_RESIDUAL_THRESHOLD)
                        status[i] = 1;
                    else
                        status[i] = 0;
                }
            }
            else
            {
                for (size_t i = 0; i < status.size(); i++)
                {
                    double y_error = status[i] ? std::abs(cur_pts[i].y - cur_right_pts[i].y) : 2.0 * STEREO_RESIDUAL_THRESHOLD;
                    right_residuals[i] = y_error;
                }
            }

            ids_right = ids;
            reduceVector(cur_right_pts, status);
            reduceVector(ids_right, status);
            reduceVector(right_residuals, status);
            // only keep left-right pts
            /*
            reduceVector(cur_pts, status);
            reduceVector(ids, status);
            reduceVector(track_cnt, status);
            reduceVector(cur_un_pts, status);
            reduceVector(pts_velocity, status);
            */
            cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
            right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);

            map<int, int> left_id_index;
            for (size_t i = 0; i < ids.size(); i++)
                left_id_index[ids[i]] = i;
            for (size_t i = 0; i < ids_right.size(); i++)
            {
                auto it = left_id_index.find(ids_right[i]);
                if (it != left_id_index.end())
                    stereo_residuals[it->second] = right_residuals[i];
            }
        }
        prev_un_right_pts_map = cur_un_right_pts_map;
    }

    for (size_t i = 0; i < ids.size(); i++)
    {
        if (!DYNAMIC_WEIGHT_ENABLE)
        {
            dynamic_weights[i] = 1.0;
            continue;
        }

        const double flow_sigma = std::max(1e-3, OPTICAL_FLOW_RESIDUAL_THRESHOLD);
        const double stereo_sigma = std::max(1e-3, STEREO_RESIDUAL_THRESHOLD);
        const double flow_weight = std::exp(-0.5 * std::pow(optical_flow_residuals[i] / flow_sigma, 2.0));
        double stereo_weight = 1.0;
        if (!_img1.empty() && stereo_cam)
            stereo_weight = std::exp(-0.5 * std::pow(stereo_residuals[i] / stereo_sigma, 2.0));

        const double track_factor = std::min(1.0, 0.5 + 0.1 * static_cast<double>(track_cnt[i]));
        double weight = flow_weight * stereo_weight;
        weight = track_factor * weight + (1.0 - track_factor) * MIN_DYNAMIC_WEIGHT;
        if (weight < DYNAMIC_WEIGHT_THRESHOLD)
            weight *= WEIGHT_DECAY_FACTOR;
        dynamic_weights[i] = std::max(MIN_DYNAMIC_WEIGHT, std::min(1.0, weight));
    }

    if(SHOW_TRACK)
        drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);

    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;
    hasPrediction = false;

    prevLeftPtsMap.clear();
    for(size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    FeatureFrame featureFrame;
    map<int, size_t> left_id_index;
    for (size_t i = 0; i < ids.size(); i++)
    {
        int feature_id = ids[i];
        double x, y ,z;
        x = cur_un_pts[i].x;
        y = cur_un_pts[i].y;
        z = 1;
        double p_u, p_v;
        p_u = cur_pts[i].x;
        p_v = cur_pts[i].y;
        int camera_id = 0;
        double velocity_x, velocity_y;
        velocity_x = pts_velocity[i].x;
        velocity_y = pts_velocity[i].y;

        FeatureObservation observation;
        observation.point = Eigen::Vector3d(x, y, z);
        observation.uv = Eigen::Vector2d(p_u, p_v);
        observation.velocity = Eigen::Vector2d(velocity_x, velocity_y);
        observation.optical_flow_residual = i < optical_flow_residuals.size() ? optical_flow_residuals[i] : 0.0;
        observation.stereo_residual = i < stereo_residuals.size() ? stereo_residuals[i] : 0.0;
        observation.dynamic_weight = i < dynamic_weights.size() ? dynamic_weights[i] : 1.0;

        featureFrame[feature_id].emplace_back(camera_id, observation);
        left_id_index[feature_id] = i;
    }

    if (!_img1.empty() && stereo_cam)
    {
        for (size_t i = 0; i < ids_right.size(); i++)
        {
            int feature_id = ids_right[i];
            double x, y ,z;
            x = cur_un_right_pts[i].x;
            y = cur_un_right_pts[i].y;
            z = 1;
            double p_u, p_v;
            p_u = cur_right_pts[i].x;
            p_v = cur_right_pts[i].y;
            int camera_id = 1;
            double velocity_x, velocity_y;
            velocity_x = right_pts_velocity[i].x;
            velocity_y = right_pts_velocity[i].y;

            FeatureObservation observation;
            observation.point = Eigen::Vector3d(x, y, z);
            observation.uv = Eigen::Vector2d(p_u, p_v);
            observation.velocity = Eigen::Vector2d(velocity_x, velocity_y);

            auto left_it = left_id_index.find(feature_id);
            if (left_it != left_id_index.end())
            {
                size_t left_index = left_it->second;
                observation.optical_flow_residual = optical_flow_residuals[left_index];
                observation.stereo_residual = stereo_residuals[left_index];
                observation.dynamic_weight = dynamic_weights[left_index];
            }

            featureFrame[feature_id].emplace_back(camera_id, observation);
        }
    }

    //printf("feature track whole time %f\n", t_r.toc());
    return featureFrame;
}

void FeatureTracker::rejectWithF()
{
    if (cur_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        const double fx = !CAM_FX.empty() ? CAM_FX.front() : FOCAL_LENGTH;
        const double fy = !CAM_FY.empty() ? CAM_FY.front() : FOCAL_LENGTH;
        const double cx = !CAM_CX.empty() ? CAM_CX.front() : col / 2.0;
        const double cy = !CAM_CY.empty() ? CAM_CY.front() : row / 2.0;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera[0]->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = fx * tmp_p.x() / tmp_p.z() + cx;
            tmp_p.y() = fy * tmp_p.y() / tmp_p.z() + cy;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera[0]->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = fx * tmp_p.x() / tmp_p.z() + cx;
            tmp_p.y() = fy * tmp_p.y() / tmp_p.z() + cy;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

void FeatureTracker::readIntrinsicParameter(const vector<string> &calib_file)
{
    m_camera.clear();
    stereo_cam = 0;
    for (size_t i = 0; i < calib_file.size(); i++)
    {
        ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);
    }
    if (calib_file.size() == 2)
        stereo_cam = 1;
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
    const double fx = !CAM_FX.empty() ? CAM_FX.front() : FOCAL_LENGTH;
    const double fy = !CAM_FY.empty() ? CAM_FY.front() : FOCAL_LENGTH;
    const double cx = !CAM_CX.empty() ? CAM_CX.front() : col / 2.0;
    const double cy = !CAM_CY.empty() ? CAM_CY.front() : row / 2.0;
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < col; i++)
        for (int j = 0; j < row; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera[0]->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * fx + cx;
        pp.at<float>(1, 0) = undistortedp[i].y() * fy + cy;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    // turn the following code on if you need
    // cv::imshow(name, undistortedImg);
    // cv::waitKey(0);
}

vector<cv::Point2f> FeatureTracker::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }
    return un_pts;
}

vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                            map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts)
{
    vector<cv::Point2f> pts_velocity;
    cur_id_pts.clear();
    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts.insert(make_pair(ids[i], pts[i]));
    }

    // caculate points velocity
    if (!prev_id_pts.empty())
    {
        double dt = cur_time - prev_time;
        
        for (unsigned int i = 0; i < pts.size(); i++)
        {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end())
            {
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            }
            else
                pts_velocity.push_back(cv::Point2f(0, 0));

        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    return pts_velocity;
}

void FeatureTracker::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts, 
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap)
{
    //int rows = imLeft.rows;
    int cols = imLeft.cols;
    if (!imRight.empty() && stereo_cam)
        cv::hconcat(imLeft, imRight, imTrack);
    else
        imTrack = imLeft.clone();
    cv::cvtColor(imTrack, imTrack, CV_GRAY2RGB);

    for (size_t j = 0; j < curLeftPts.size(); j++)
    {
        double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
        cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    if (!imRight.empty() && stereo_cam)
    {
        for (size_t i = 0; i < curRightPts.size(); i++)
        {
            cv::Point2f rightPt = curRightPts[i];
            rightPt.x += cols;
            cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            //cv::Point2f leftPt = curLeftPtsTrackRight[i];
            //cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
        }
    }
    
    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++)
    {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if(mapIt != prevLeftPtsMap.end())
        {
            cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }

    //draw prediction
    /*
    for(size_t i = 0; i < predict_pts_debug.size(); i++)
    {
        cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
    }
    */
    //printf("predict pts size %d \n", (int)predict_pts_debug.size());

    //cv::Mat imCur2Compress;
    //cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}


void FeatureTracker::setPrediction(map<int, Eigen::Vector3d> &predictPts)
{
    hasPrediction = true;
    predict_pts.clear();
    predict_pts_debug.clear();
    map<int, Eigen::Vector3d>::iterator itPredict;
    for (size_t i = 0; i < ids.size(); i++)
    {
        //printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
        int id = ids[i];
        itPredict = predictPts.find(id);
        if (itPredict != predictPts.end())
        {
            Eigen::Vector2d tmp_uv;
            m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);
            predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
            predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
        }
        else
            predict_pts.push_back(prev_pts[i]);
    }
}


void FeatureTracker::removeOutliers(set<int> &removePtsIds)
{
    std::set<int>::iterator itSet;
    vector<uchar> status;
    for (size_t i = 0; i < ids.size(); i++)
    {
        itSet = removePtsIds.find(ids[i]);
        if(itSet != removePtsIds.end())
            status.push_back(0);
        else
            status.push_back(1);
    }

    reduceVector(prev_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
    reduceVector(optical_flow_residuals, status);
    reduceVector(stereo_residuals, status);
    reduceVector(dynamic_weights, status);
}


cv::Mat FeatureTracker::getTrackImage()
{
    return imTrack;
}
