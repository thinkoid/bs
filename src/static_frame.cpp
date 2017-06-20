#include <bs/static_frame.hpp>
#include <bs/utils.hpp>

#include <opencv2/imgproc.hpp>

namespace bs {

/* explicit */
static_frame::static_frame (
    const cv::Mat& background, size_t threshold /* = 15 */)
    : background_ (background), threshold_ (threshold)
{ }

const cv::Mat&
static_frame::operator() (const cv::Mat& frame) {
    return mask_ = threshold (absdiff (frame, background_), threshold_);
}

}
