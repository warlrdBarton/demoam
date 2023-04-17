#include "feature_tracker.h"
#include "frame.h"
#include "feature.h"
#include "mappoint.h"
#include "camera.h"

namespace demoam {
    int FeatureTracker::DetectFastInLeft(std::shared_ptr<Frame> frame) {
        if (fast_detector_ == nullptr) {
            fast_detector_ = cv::FastFeatureDetector::create(10, true);
        }

        cv::Mat mask(frame -> img_left_.size(), CV_8UC1, 255);
        for (auto& feat : frame -> features_left_) {
            cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                          feat->position_.pt + cv::Point2f(10, 10), 0, cv::FILLED);
        }

        std::vector<cv::KeyPoint> keypoints;
        fast_detector_ -> detect(frame -> img_left_, keypoints, mask);
        int cntDetected = 0;
        for (auto& kp : keypoints) {
            frame -> features_left_.push_back(
                std::shared_ptr<Feature>(new Feature(frame, kp))
            );
            cntDetected++;
        }
        return cntDetected;
    }

    int FeatureTracker::SearchLastFrameOpticalFlow(std::shared_ptr<Frame> last_frame, std::shared_ptr<Frame> current_frame, std::shared_ptr<demoam::Camera> camera_left, int f_threshold) {
        std::vector<cv::Point2f> kps_last, kps_current; 
        for (auto& kp : last_frame -> features_left_) {
            kps_last.push_back(kp -> position_.pt);
            auto mp = kp -> mappoint_.lock();
            if (mp) {
                auto px = camera_left -> world2pixel(mp -> pos_, current_frame -> Pose());
                kps_current.push_back(cv::Point2f(px[0], px[1]));
            } else {
                kps_current.push_back(kp -> position_.pt);
            }
        }
        std::vector<uchar> status;
        std::vector<float> error;
        cv::calcOpticalFlowPyrLK(last_frame -> img_left_, current_frame -> img_left_, 
                                 kps_last, kps_current, status, error, cv::Size(11, 11), 1
        );
        //cv::findFundamentalMat(kps_current, kps_last, cv::FM_RANSAC, f_threshold, 0.99, status);
       
        int num_good_pts = 0;
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i]) {
                cv::KeyPoint kp(kps_current[i], 7);
                std::shared_ptr<Feature> new_feature(new Feature(current_frame, kp));
                new_feature -> mappoint_ = last_frame -> features_left_[i] -> mappoint_;
                current_frame -> features_left_.push_back(new_feature);
                num_good_pts++;
            }
        }
        LOG(INFO) << "FeatureTracker::TrackLastFrameOpticalFlow: " << num_good_pts << " keypoints tracked successfuly from LastFrame";
        return num_good_pts;
    }

    int FeatureTracker::SearchInRightOpticalFlow(std::shared_ptr<Frame> current_frame,  std::shared_ptr<demoam::Camera> camera_right, int f_threshold) {
        std::vector<cv::Point2f> kps_left, kps_right;
        for (auto& kp : current_frame -> features_left_) {
            kps_left.push_back(kp -> position_.pt);
            auto mp = kp -> mappoint_.lock();
            if (mp) {
                auto px = camera_right -> world2pixel(mp -> pos_, current_frame -> Pose());
                kps_right.push_back(cv::Point2f(px[0], px[1]));
            } else {
                kps_right.push_back(kp -> position_.pt);
            }
        }

        std::vector<uchar> status;
        std::vector<float> error;
        cv::calcOpticalFlowPyrLK(current_frame -> img_left_, current_frame -> img_right_, 
                                 kps_left, kps_right, status, error, cv::Size(11, 11), 1
        );
        //cv::findFundamentalMat(kps_right, kps_left, cv::FM_RANSAC, f_threshold, 0.99, status);

        int num_good_pts = 0;
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i]) {
                cv::KeyPoint kp(kps_right[i], 7);
                std::shared_ptr<Feature> new_feature(new Feature(current_frame, kp));
                new_feature -> is_on_left_image_ = false;
                current_frame -> features_right_.push_back(new_feature);
                num_good_pts++;
            } else {
                current_frame -> features_right_.push_back(nullptr);
            }
        }
        return num_good_pts;
    }


    std::shared_ptr<cv::FastFeatureDetector> FeatureTracker::fast_detector_ = nullptr;
}