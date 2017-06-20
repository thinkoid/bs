#include <bs/utils.hpp>
#include <bs/adaptive_learning.hpp>

#include <opencv2/imgproc.hpp>

namespace bs {

/* explicit */
adaptive_learning::adaptive_learning (
    const cv::Mat& background, double alpha, size_t threshold)
    : background_ (background), alpha_ (alpha), threshold_ (threshold)
{ }

const cv::Mat&
adaptive_learning::operator() (const cv::Mat& frame) {
    cv::Mat diff = absdiff (frame, background_);

    //
    // Unlimited integration of the incoming frames into the background:
    //
    background_ = alpha_ * frame + (1 - alpha_) * background_;

    return mask_ = threshold (diff, threshold_);
}

}
