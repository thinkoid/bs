#include <bs/windowed_moving_mean.hpp>
#include <bs/utils.hpp>

#include <opencv2/imgproc.hpp>

namespace bs {

/* explicit */
windowed_moving_mean::windowed_moving_mean (
    std::vector< double > weights, int threshold)
    : framebuf_ (3), weights_ (weights), threshold_ (threshold)
{ }

const cv::Mat&
windowed_moving_mean::operator() (const cv::Mat& frame) {
    framebuf_.push_back (float_from (frame));

    if (framebuf_.size () < 3)
        return mask_ = mono_from (frame);

    cv::Mat background = mono_from (
        framebuf_ [0] * weights_ [0] +
        framebuf_ [1] * weights_ [1] +
        framebuf_ [2] * weights_ [2]);

    return mask_ = threshold (absdiff (frame, background), threshold_);
}

}
