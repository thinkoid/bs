#include <bs/utils.hpp>
#include <bs/adaptive_median.hpp>

#include <opencv2/imgproc.hpp>

namespace bs {

/* explicit */
adaptive_median::adaptive_median (
    const cv::Mat& background, size_t frame_interval, size_t threshold)
    : background_ (background), mask_ (),
      frame_interval_ (frame_interval), frame_counter_ (),
      threshold_ (threshold) {
}

//
// ... Image differencing between the current frame and a reference image gave
// better segmentation results than differencing between successive frames
// because it did not produce false positives where a dark shadow had moved away
// from an area of background. It identified the whole area of the piglet,
// rather than just the leading edge, and it was able to locate piglets that
// were not currently moving ...
//
// The reference image was a running median of the image sequence produced by
// the following method. Each pixel in the reference image was incremented by
// one if the corresponding pixel in the current image was greater in value or
// decreased by one if the current image pixel was less in value. Each pixel in
// the reference image then converged to a value for which half the updating
// values were greater than and half were less than this value - that is, the
// median. This technique requires the storage of only one reference image, and
// is computationally inexpensive. The median was chosen in preference to the
// mean because of its better rejection of outliers in the distribution of pixel
// values.
//

const cv::Mat&
adaptive_median::operator() (const cv::Mat& frame) {
    //
    // The motion mask is a simple thresholded difference between the current
    // frame and the reference (background) frame:
    //
    mask_ = threshold (absdiff (frame, background_), threshold_);

    if (0 == frame_counter_++ % frame_interval_) {
        //
        // Update the reference (background) frame:
        //
        cv::Mat add_mask = threshold (frame - background_, 0, 1);
        cv::Mat sub_mask = threshold (background_ - frame, 0, 1);

        background_ = background_ + add_mask - sub_mask;
    }

    return mask_;
}

}
