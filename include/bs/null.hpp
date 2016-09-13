#ifndef BS_NULL_HPP
#define BS_NULL_HPP

#include <bs/defs.hpp>
#include <opencv2/core/mat.hpp>

namespace bs {

struct null {
    const cv::Mat&
    operator() (const cv::Mat& arg) {
        return frame_ = arg;
    }

    const cv::Mat&
    mask () const {
        return frame_;
    }

private:
    cv::Mat frame_;
};

}

#endif // BS_NULL_HPP
