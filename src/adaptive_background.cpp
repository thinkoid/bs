#include <bs/utils.hpp>
#include <bs/adaptive_background.hpp>

#include <opencv2/imgproc.hpp>

namespace bs {

/* explicit */
adaptive_background::adaptive_background (
    const cv::Mat& background, double alpha, int threshold)
    : background_ (background), alpha_ (alpha), threshold_ (threshold)
{ }

const cv::Mat&
adaptive_background::operator() (const cv::Mat& frame) {
    cv::Mat diff = absdiff (frame, background_);

    //
    // Continuously integrate the incoming frames into the background:
    //
    background_ = alpha_ * frame + (1 - alpha_) * background_;

    return mask_ = threshold (diff, threshold_);
}

}
