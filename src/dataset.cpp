#include "dataset.h"

#include <boost/format.hpp>
#include <glog/logging.h>

#include "camera.h"
#include "frame.h"
#include "imu_types.h"
#include "geometric_tools.h"

namespace demoam {
    
Dataset::Dataset(const std::string& dataset_path, const std::string& imu_path) 
    : dataset_path_(dataset_path), 
      imu_path_(imu_path) {}

bool Dataset::Init() {
    LOG(INFO) << "Dataset::Init(): Loading calibrations" <<"...";
    LoadCalib();
    LOG(INFO) << "Dataset::Init(): LOADED";

    LOG(INFO) << "Dataset::Init(): Loading images for sequence " <<"...";
    LoadImages();
    LOG(INFO) << "Dataset::Init(): LOADED";

    LOG(INFO) << "Dataset::Init(): Loading IMU for sequence " <<"...";
    LoadIMU();
    LOG(INFO) << "Dataset::Init(): LOADED";

    if (vTimestampsCam_.empty() || vTimestampsImu_.empty() || vAcc_.empty() || vGyro_.empty()) {
        LOG(ERROR) << "Dataset::Init(): Failed to load images or IMU for sequence";
        return false;
    }

    // Find first imu to be considered, supposing imu measurements start first
    while(vTimestampsImu_[first_imu_] <= vTimestampsCam_[0]) {
        first_imu_++;
    }
    first_imu_--; // first imu measurement to be considered

    return true;
}

bool Dataset::LoadCalib() {
    std::ifstream fin(dataset_path_ + "/calib.txt");
    if (!fin) {
        LOG(ERROR) << "Could not Find " << dataset_path_ << "/calib.txt!";
        return false;
    }

    Sophus::SE3d Tvi, Tiv; // per Kitti calibration
    {
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        R << 9.999976e-01, 7.553071e-04, -2.035826e-03, 
            -7.854027e-04, 9.998898e-01, -1.482298e-02,
            2.024406e-03, 1.482454e-02, 9.998881e-01;
        R = NormalizeRotation(R);
        t << -8.086759e-01, 3.195559e-01, -7.997231e-01;
        Tvi = Sophus::SE3d(R, t);
        Tiv = Tvi.inverse();
    }

    Sophus::SE3d Tcv, Tvc;
    {
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        R << 7.027555e-03, -9.999753e-01, 2.599616e-05,
            -2.254837e-03, -4.184312e-05, -9.999975e-01,
             9.999728e-01, 7.027479e-03, -2.255075e-03;
        R = NormalizeRotation(R);
        t << -7.137748e-03, -7.482656e-02, -3.336324e-01;
        Tcv = Sophus::SE3d(R, t);
        Tvc = Tcv.inverse();
    }

    Sophus::SE3d Tci, Tci_inv;
    Tci = Tcv * Tvi;
    Tci_inv = Tci.inverse();

    for (int i = 0; i < 4; ++i) {
        char camera_name[3];
        for (int k = 0; k < 3; ++k) {
            fin >> camera_name[k];
        }
        double projection_data[12];
        for (int k = 0; k < 12; ++k) {
            fin >> projection_data[k];
        }
        Eigen::Matrix3d K;
        K << projection_data[0], projection_data[1], projection_data[2],
            projection_data[4], projection_data[5], projection_data[6],
            projection_data[8], projection_data[9], projection_data[10];
        Eigen::Vector3d t;
        t << projection_data[3], projection_data[7], projection_data[11];

        t = K.inverse() * t; // from pixels to metres
        K = K * 0.5; // for later down sample

        Sophus::SE3d pose = Sophus::SE3d(Sophus::SO3d(), t);

        std::shared_ptr<Camera> new_camera(new Camera(K(0, 0), K(1, 1), K(0, 2), K(1, 2), 
                                                    t.norm(), pose                                              
        ));
        new_camera->tci = Tci;
        new_camera->tci_inv = Tci_inv;
        cameras_.push_back(new_camera);
        
        LOG(INFO) << "Dataset::Init(): Camera " << i << " extrinsics: " << t.transpose();
    }
    fin.close();
    return true;
}

bool Dataset::LoadImages() {
    std::ifstream fin(dataset_path_ + "/times.txt");
    if (!fin) {
        LOG(ERROR) << "Could not Find " << dataset_path_ << "/times.txt!";
        return false;
    }
    while (!fin.eof()) {
        std::string str;
        getline(fin, str);
        if (!str.empty()) {
            std::stringstream ss(str);
            double timestamp;
            ss >> timestamp;
            vTimestampsCam_.push_back(timestamp);
        }
    }
    fin.close();
    return true;
}

bool Dataset::LoadIMU() {
    std::ifstream fTimes(imu_path_ + "/times.txt");
    if (!fTimes) {
        LOG(ERROR) << "Could not Find " << imu_path_ << "/times.txt!";
        return false;
    }
    vTimestampsImu_.reserve(100000);
    while (!fTimes.eof()) {
        std::string s;
        getline(fTimes, s);
        if(!s.empty())
        {
            std::stringstream ss(s);
            double timestamp;
            ss >> timestamp;
            vTimestampsImu_.push_back(timestamp);
        }        
    }
    fTimes.close();
    LOG(INFO) << "Dataset::LoadIMU(): Loaded IMU timestamps";

    const int nTimes = vTimestampsImu_.size();

    for (int i = 0; i < nTimes; ++i) {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(10) << i;
        std::string pathtemp = imu_path_ + "/data/" + ss.str() + ".txt";
        std::ifstream fin (pathtemp);
        if (!fin)
        {
            LOG(ERROR) << "Could not Find IMU data at path " << pathtemp;
            return false;
        }

        std::string s;
        getline(fin, s);
        if(!s.empty())
        {
            std::string item;
            size_t pos = 0;
            double data[23];
            int count = 0;
            while ((pos = s.find(' ')) != std::string::npos) {
                item = s.substr(0, pos);
                data[count++] = stod(item);
                s.erase(0, pos + 1);
            }
            item = s.substr(0, pos);
            data[6] = stod(item);

            vAcc_.push_back(cv::Point3f(data[11],data[12],data[13]));
            vGyro_.push_back(cv::Point3f(data[17],data[18],data[19]));
        }
    }
    return true;
}

std::shared_ptr<Frame> Dataset::NextFrame() {
    boost::format fmt("%s/image_%d/%06d.png");
    cv::Mat image_left, image_right;
    image_left = cv::imread((fmt % dataset_path_ % 0 %current_image_index_).str(),
                            cv::IMREAD_GRAYSCALE);
    image_right = cv::imread((fmt % dataset_path_ % 1 %current_image_index_).str(),
                            cv::IMREAD_GRAYSCALE);

    if (image_left.data == nullptr || image_right.data == nullptr) {
        LOG(WARNING) << "Could not read the images at index " << current_image_index_;
        return nullptr;
    }

    cv::Mat image_left_resized, image_right_resized;
    cv::resize(image_left, image_left_resized, cv::Size(), 0.5, 0.5,
               cv::INTER_NEAREST);
    cv::resize(image_right, image_right_resized, cv::Size(), 0.5, 0.5,
               cv::INTER_NEAREST);
    

    // Load imu measurements from previous frame
    std::vector<IMU, Eigen::aligned_allocator<IMU>> imu_meas_since_last_frame;
    if (current_image_index_ > 0) {
        while (vTimestampsImu_[first_imu_] <= vTimestampsCam_[current_image_index_]) {
            imu_meas_since_last_frame.push_back(IMU(vTimestampsImu_[first_imu_], 
                                                    vGyro_[first_imu_].x, vGyro_[first_imu_].y, vGyro_[first_imu_].z,
                                                    vAcc_[first_imu_].x, vAcc_[first_imu_].y, vAcc_[first_imu_].z
                                                ));
            first_imu_++;
        }
    }

    std::shared_ptr<Frame> new_frame = Frame::CreateFrame();
    new_frame -> img_left_ = image_left_resized;
    new_frame -> img_right_ = image_right_resized;
    new_frame -> time_stamp_ = vTimestampsCam_[current_image_index_];
    new_frame -> imu_meas_ = imu_meas_since_last_frame; 

    current_image_index_++;
    return new_frame;
}

} // namespace demoam

