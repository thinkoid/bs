#ifndef BS_ADAPTIVE_MEDIAN_HPP
#define BS_ADAPTIVE_MEDIAN_HPP

#include <opencv2/core/mat.hpp>

namespace bs {

struct adaptive_median {
    explicit adaptive_median (const cv::Mat&, size_t, int);

public:
    const cv::Mat&
    operator() (const cv::Mat&);

    const cv::Mat&
    mask () const {
        return mask_;
    }

private:
    cv::Mat background_, mask_;
    size_t frame_interval_, frame_counter_;
    int threshold_;
};

}

#endif // BS_ADAPTIVE_MEDIAN_HPP
