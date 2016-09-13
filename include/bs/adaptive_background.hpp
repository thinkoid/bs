#ifndef BS_ADAPTIVE_BACKGROUND_HPP
#define BS_ADAPTIVE_BACKGROUND_HPP

#include <opencv2/core/mat.hpp>

namespace bs {

struct adaptive_background {
    explicit adaptive_background (const cv::Mat&, double, int);

public:
    const cv::Mat&
    operator() (const cv::Mat&);

    const cv::Mat
    mask () const {
        return mask_;
    }

private:
    cv::Mat background_, mask_;

    double alpha_;
    int threshold_;
};

}

#endif // BS_ADAPTIVE_BACKGROUND_HPP
