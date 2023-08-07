#ifndef DEMOAM__CONFIG_H
#define DEMOAM__CONFIG_H

#include "common_include.h"

namespace demoam {

class Config {
 public:
    ~Config();
    static bool SetParameterFile(const std::string& filename);
    template <typename T> 
    static T Get(const std::string &key) {
        return T(Config::config_ -> file_[key]);
    }
 private:
    static std::shared_ptr<Config> config_;
    cv::FileStorage file_;
    Config() {} // private constructor makes a singleton
};

} // namespace demoam

#endif // DEMOAM__CONFIG_H