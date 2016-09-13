#include <bs/utils.hpp>
#include <bs/adaptive_selective_background.hpp>

#include <opencv2/imgproc.hpp>

namespace bs {

/* explicit */
adaptive_selective_background::adaptive_selective_background (
    const cv::Mat& background, double alpha, int threshold)
    : background_ (background), mask_ (background.size (), CV_8U),
      alpha_ (alpha), threshold_ (threshold)
{ }

const cv::Mat&
adaptive_selective_background::operator() (const cv::Mat& frame) {
    cv::Mat diff = absdiff (frame, background_);

    cv::Mat foreground_mask = threshold (
        median_blur (threshold (diff, threshold_), 3), 0, 1);

    cv::Mat background_mask = 1 - foreground_mask;

    background_ =
        //
        // Keep the foreground portion of the background unchanged:
        //
        multiply (background_, foreground_mask) +

        //
        // Weigh the background portion of the background and the background
        // portion of the current frame:
        //
        (multiply (frame, background_mask) * alpha_ +
         multiply (background_, background_mask) * (1. - alpha_));

    return mask_ = 255 * foreground_mask;
}

}
