# DEMOAM-Stereo-VIO
This is DEMOAM-Stereo-VIO, a stereo inertial VO code.
It uses a LK optical flow as front-end and a sliding window bundle adjustment as a backend.

# Dependencies
- Pangolin (for visualization): https://github.com/stevenlovegrove/Pangolin
- Eigen3: sudo apt-get install libeigen3-dev
- g2o: sudo apt-get install libcxsparse-dev libqt4-dev libcholmod3.0.6 libsuitesparse-dev qt4-qmake
- OpenCV: sudo apt-get install libopencv-dev
- glog (for logging): sudo apt-get install libgoogle-glog-dev

# Compile
run "./recmake.sh" and "./make_run_eva.sh" to compile and run all the things, or follow the steps in such.

# Examples
The vio now support only KITTI datasets format.
Before Running, go check"config/default.yaml", specify data directories and tweak related parameters.
Then, run with one line code as below. 
```
./bin/stereo_kitti
```

# About
DEMOAM means "Demo odometry and mapping", which is author's first work in visual SLAM. It runs fast and smooth in road-driving scenarios, though struggles facing complicated cases, which is an inevitable regarding its neat design and clear codes. Thanks to the inspirations of many spectacular vo frameworks, like ORB-SLAM by SLAMLab - Universidad de Zaragozaby and VINS by TONG, QIN.


