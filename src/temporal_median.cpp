#include <iostream>

#include <bs/utils.hpp>
#include <bs/temporal_median.hpp>

namespace bs {

temporal_median_t::temporal_median_t (
    const cv::Mat& b, size_t h, size_t i, size_t lo, size_t hi)
    : detail::base_t (b.clone (), { b.size (), CV_8U }), history_ (h),
    lo_ (lo), hi_ (hi), frame_interval_ (i), frame_counter_ { }
{ }

cv::Mat
temporal_median_t::calculate_median () const {
    cv::Mat median (background_.size (), CV_8U);

    const auto n = history_.size () + 1;
    std::vector< unsigned char > buf (n);

    for (size_t i = 0; i < median.total (); ++i) {
        //
        // Extract pixel history from the historic frames:
        //
        std::transform (
            history_.begin (), history_.end (), buf.begin (), [&](auto& x) {
                return x.template at< unsigned char > (i);
            });

        //
        // Use the current background, i.e., the median from the previous
        // iteration:
        //
        buf.back () = background_.at< unsigned char > (i);

        //
        // Sort the set of historic pixels and current background pixel at the
        // position based on their gray levels:
        //
        std::sort (buf.begin (), buf.end ());

        //
        // The median is the new background:
        //
        median.at< unsigned char > (i) = buf [n / 2];
    }

    return median;
}

cv::Mat
temporal_median_t::merge_masks (const cv::Mat& lo_mask, const cv::Mat& hi_mask) {
    const unsigned char *p = lo_mask.data, *q = hi_mask.data;

    const size_t w = lo_mask.cols, h = lo_mask.rows;
    cv::Mat mask (h, w, CV_8U, cv::Scalar (0));

    unsigned char* r = mask.data;

    for (size_t i = 1; i < h - 1; ++i) {
        for (size_t j = 1; j < w - 1; ++j) {

            const size_t pos = i * w + j;
            BS_ASSERT (pos < w * h);

            //
            // A pixel is marked as foreground ... if it is presented(sic) in
            // the low-thresholded binarized mask AND it is spatially connected
            // to at least one pixel present in the high-thresholded binarized
            // mask:
            //
            if (q [pos] || p [pos] && (
                    q [pos - w - 1] ||
                    q [pos - w] ||
                    q [pos - w + 1] ||
                    q [pos - 1] ||
                    q [pos + 1] ||
                    q [pos + w - 1] ||
                    q [pos + w] ||
                    q [pos + w + 1])) {
                r [pos] = 255;
            }
        }
    }

    return mask;
}

const cv::Mat&
temporal_median_t::operator () (const cv::Mat& frame) {
    if (history_.size () < history_.capacity ()) {
        //
        // Store frames until the history buffer is full:
        //
        return history_.push_back (frame), mask_ = frame;
    }
    else {
        //
        // Compute the mask -- as a plain absolute difference between the
        // incoming frame and background:
        //
        cv::Mat diff = absdiff (frame, background_);

        auto median = calculate_median ();
        median.copyTo (background_);

        if (0 == (++frame_counter_ % frame_interval_))
            history_.push_back (frame);

        //
        // Create two masks: a "low threshold" mask and a "high threshold"
        // mask. This approach uses a pair of simpler, global threshold masks,
        // than the algorithm described in the cited paper:
        //
        return mask_ = merge_masks (
            threshold (diff, lo_), threshold (diff, hi_));
    }
}

}
