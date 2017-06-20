#include <bs/simple_frame.hpp>
#include <bs/utils.hpp>

#include <opencv2/imgproc.hpp>

#include <iterator>

namespace bs {

/* explicit */
simple_frame::simple_frame (const cv::Mat& prev, size_t threshold)
    : background_ (prev), threshold_ (threshold)
{ }

const cv::Mat&
simple_frame::operator() (const cv::Mat& frame) {
    mask_ = threshold (absdiff (frame, background_), threshold_);
    background_ = frame;
    return mask_;
}

}
