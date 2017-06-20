#ifndef BS_MOVING_MEAN_HPP
#define BS_MOVING_MEAN_HPP

#include <bs/defs.hpp>
#include <opencv2/core/mat.hpp>

namespace bs {

struct moving_mean {
    explicit moving_mean (const cv::Mat&, double, size_t);

public:
    const cv::Mat&
    operator() (const cv::Mat&);

    const cv::Mat&
    mask () const {
        return mask_;
    }

private:
    cv::Mat mean_, mask_;
    double alpha_;
    size_t threshold_;
};

}

#endif // BS_MOVING_MEAN_HPP
