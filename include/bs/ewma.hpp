#ifndef BS_EWMA_HPP
#define BS_EWMA_HPP

#include <bs/defs.hpp>
#include <opencv2/core/mat.hpp>

namespace bs {

struct ewma_t {
    explicit ewma_t (double alpha = .05) : alpha_ (alpha) { }

public:
    cv::Mat operator() (const cv::Mat& src) {
        if (value_.empty ()) {
            value_ = src;
        }
        else {
            value_ = alpha_ * src + (1 - alpha_) * value_;
        }

        return value_;
    }

    const cv::Mat& value () const {
        return value_;
    }

private:
    cv::Mat value_;
    double alpha_;
};

}

#endif // BS_EWMA_HPP
