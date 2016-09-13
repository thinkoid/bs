#include <bs/utils.hpp>
#include <bs/moving_mean.hpp>

#include <opencv2/imgproc.hpp>

namespace bs {

/* explicit */
moving_mean::moving_mean (
    const cv::Mat& first, double alpha, int threshold)
    : mean_ (float_from (first)),
      mask_ (first.size (), CV_8U),
      alpha_ (alpha), threshold_ (threshold) {
}

const cv::Mat&
moving_mean::operator() (const cv::Mat& frame) {
    cv::Mat fframe = float_from (frame);

    cv::Mat dist = power_of (absdiff (fframe, mean_), 2);
    mean_ = alpha_ * mean_ + (1. - alpha_) * fframe;

    mask_ = threshold (absdiff (frame, mono_from (mean_)), threshold_);

    return mask_;
}

}
