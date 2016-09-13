#ifndef BS_MOVING_MEAN_HPP
#define BS_MOVING_MEAN_HPP

#include <bs/defs.hpp>
#include <opencv2/core/mat.hpp>

namespace bs {

struct moving_mean {
    explicit moving_mean (const cv::Mat&, double, int);

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
    int threshold_;
};

}

#endif // BS_MOVING_MEAN_HPP
