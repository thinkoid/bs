#ifndef BS_ADAPTIVE_SELECTIVE_BACKGROUND_HPP
#define BS_ADAPTIVE_SELECTIVE_BACKGROUND_HPP

#include <bs/defs.hpp>
#include <opencv2/core/mat.hpp>

namespace bs {

struct adaptive_selective_background {
    explicit adaptive_selective_background (
        const cv::Mat&, double, int);

public:
    const cv::Mat&
    operator() (const cv::Mat&);

    const cv::Mat&
    mask () const {
        return mask_;
    }

private:
    cv::Mat background_, mask_;
    double alpha_;
    int threshold_;
};

}

#endif // BS_ADAPTIVE_SELECTIVE_BACKGROUND_HPP
