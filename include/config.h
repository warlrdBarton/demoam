#ifndef DEMOAM__CONFIG_H
#define DEMOAM__CONFIG_H

#include "common_include.h"

namespace demoam {

class Config {
 private:
    static std::shared_ptr<Config> config_;
    cv::FileStorage file_;
    Config() {} // private constructor makes a singleton
 public:
    ~Config();
    static bool SetParameterFile(const std::string& filename);
    template <typename T> 
    static T Get(const std::string &key) {
        return T(Config::config_ -> file_[key]);
    }
};

} // namespace demoam

#endif // DEMOAM__CONFIG_H