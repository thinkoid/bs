#include <bs/previous_frame.hpp>
#include <bs/utils.hpp>

#include <opencv2/imgproc.hpp>

#include <iterator>

namespace bs {

/* explicit */
previous_frame::previous_frame (const cv::Mat& prev, int threshold)
    : background_ (prev), threshold_ (threshold)
{ }

const cv::Mat&
previous_frame::operator() (const cv::Mat& frame) {
    mask_ = threshold (absdiff (frame, background_), threshold_);
    background_ = frame;
    return mask_;
}

}
