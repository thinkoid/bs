#ifndef BS_ADAPTIVE_SELECTIVE_LEARNING_HPP
#define BS_ADAPTIVE_SELECTIVE_LEARNING_HPP

#include <bs/defs.hpp>
#include <opencv2/core/mat.hpp>

namespace bs {

struct adaptive_selective_learning {
    explicit adaptive_selective_learning (const cv::Mat&, double, size_t);

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
    size_t threshold_;
};

}

#endif // BS_ADAPTIVE_SELECTIVE_LEARNING_HPP
