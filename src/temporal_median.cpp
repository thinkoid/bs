#include <iostream>

#include <bs/utils.hpp>
#include <bs/temporal_median.hpp>

#include <boost/range/adaptor/sliced.hpp>
#include <boost/range/adaptor/strided.hpp>
#include <boost/range/algorithm/copy.hpp>

namespace bs {

/* explicit */
temporal_median_bootstrap::temporal_median_bootstrap (
    size_t block_size,
    size_t threshold,
    size_t threshold_increment,
    size_t motionthreshold,
    size_t stablethreshold,
    size_t thrash_limit)
    : framebuf_ (2),
      block_size_ (block_size),
      threshold_ (threshold),
      threshold_increment_ (threshold_increment),
      motionthreshold_ (motionthreshold),
      stablethreshold_ (stablethreshold),
      thrashed_frames_ (),
      thrash_limit_ (thrash_limit),
      init_ (),
      complete_ ()
{ }

bool
temporal_median_bootstrap::initialize_background_from (const cv::Mat& frame) {
    background_ = frame.clone ();
    framebuf_.push_back (frame.clone ());

    const size_t n = block_size_;

    std::vector< size_t > w (frame.cols / n, n), h (frame.rows / n, n);

    if (const size_t rest = frame.cols % n)
        w.push_back (rest);

    if (const size_t rest = frame.rows % n)
        h.push_back (rest);

    size_t i = 0;

    for (const auto r : w) {
        size_t j = 0;

        for (const auto c : h) {
            const size_t motionthreshold = size_t (
                (n * n) * (double (motionthreshold_) / 100));

            //
            // Our approach basically partitions the image into blocks (of 16×16
            // pixels) and selectively updates the background model with a block
            // whenever a suﬃciently high number  of pixels within the block are
            // not in motion [...]  If more than 95% of the  pixels in the block
            // are detected as not in motion,  the model is updated in that area
            // with the pixels  not in motion. If this happens  for more than 10
            // times (also  non consecutive),  the whole  block is  considered
            // “stable” and no more evaluated.
            //
            blocks_.emplace_back (
                block { i * n, j * n, c, r, 0, motionthreshold, false });

            ++j;
        }

        ++i;
    }

    return complete_;
}

bool
temporal_median_bootstrap::update_background_from (const cv::Mat& frame) {
    framebuf_.push_back (frame.clone ());
    assert (framebuf_.size () == framebuf_.capacity ());

    const cv::Mat &p = framebuf_ [0], &q = framebuf_ [1];

    //
    // Calculate thresholded simple frame difference between two consecutive
    // frames:
    //
    cv::Mat mask = threshold (absdiff (p, q), threshold_, 1);

    size_t stabilized_blocks = 0;

    for (auto& b : blocks_) {

        if (b.stable)
            //
            // Only process unstable ROIs:
            //
            continue;

        cv::Rect r (b.x, b.y, b.w, b.h);
        cv::Mat roi (mask, r);

        const size_t n = cv::countNonZero (roi);

        if (n < b.motionthreshold) {
            {
                //
                // A block is stable and will contribute to the background model
                // if less than `motionthreshold_' pixels are stable. A pixel is
                // stable if it changes less than `threshold_' in between two
                // frames. The stable pixels in the block are copied to the
                // background model.
                //
                cv::Mat from = cv::Mat (q, r), to = cv::Mat (background_, r);
                from.copyTo (to, 1 - roi);
            }

            if (++b.updates >= stablethreshold_) {
                //
                // A block updated stablethreshold_ or more times (but not
                // necessarily consecutive!) is considered stable:
                //
                b.stable = true;

                //
                // Count stabilized ROIs:
                //
                ++stabilized_blocks;

                //
                // Reset the counter of thrashed (in motion) frames:
                //
                thrashed_frames_ = 0;
            }
        }
    }

    if (0 == stabilized_blocks) {
        if (++thrashed_frames_ > thrash_limit_) {
            //
            // Increase threshold to avoid deadlocks and speed up
            // bootstrapping:
            //
            if (255 < (threshold_ += threshold_increment_))
                threshold_ = 255;

            thrashed_frames_ = 0;
        }
    }

    return complete_ = all_of (
        blocks_.begin (), blocks_.end (), [](const auto& b) {
            return b.stable;
        });
}

bool
temporal_median_bootstrap::process (const cv::Mat& frame) {
    return 0 == init_ && 1 == ++init_
        ? initialize_background_from (frame)
        : update_background_from (frame);
}

////////////////////////////////////////////////////////////////////////

temporal_median::temporal_median (
    const cv::Mat& background,
    size_t history_size, size_t frame_interval,
    double lambda, size_t lo, size_t hi)
    : background_ (background.clone ()),
      mask_ (background.size (), CV_8U),
      history_ (history_size),
      lambda_ (lambda), lo_ (lo), hi_ (hi),
      frame_interval_ (frame_interval),
      frame_counter_ () {
    assert (background_.channels () == 1);
}

std::tuple< cv::Mat, cv::Mat, cv::Mat >
temporal_median::calculate_masks () const {
    cv::Mat background (background_.size (), CV_8U);
    cv::Mat lo_mask    (background_.size (), CV_8U);
    cv::Mat hi_mask    (background_.size (), CV_8U);

    std::vector< unsigned char > buf (history_.size () + 1);
    const size_t median_pos = (history_.size () + 1) / 2;

    for (size_t i = 0; i < background_.total (); ++i) {
        //
        // Make a copy of the current pixel history because we wish to preserve
        // the history order:
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
        // The low threshold mask value is the difference between pixels closer
        // in position, hence in value, to the median:
        //
        lo_mask.data [i] = cv::saturate_cast< unsigned char > (
            lambda_ * std::abs (int (buf [median_pos + lo_]) -
                                int (buf [median_pos - lo_])));

        //
        // The high threshold mask value is the difference between pixels farther
        // apart, hence in value, from the median:
        //
        hi_mask.data [i] = cv::saturate_cast< unsigned char > (
            lambda_ * std::abs (int (buf [median_pos + hi_]) -
                                int (buf [median_pos - hi_])));

        //
        // Update the background to the new median value:
        //
        background.at< unsigned char > (i) = buf [median_pos];
    }

    return { lo_mask, hi_mask, background };
}

cv::Mat
temporal_median::compose_masks (
    const cv::Mat& lo_mask, const cv::Mat& hi_mask, size_t maxval) {

    assert (lo_mask.size () == hi_mask.size ());
    assert (lo_mask.type () == hi_mask.type ());

    const unsigned char *p = lo_mask.data, *q = hi_mask.data;

    const size_t w = lo_mask.cols, h = lo_mask.rows;
    cv::Mat mask (h, w, CV_8U, cv::Scalar (0));

    unsigned char* r = mask.data;

    for (size_t i = 1; i < h - 1; ++i) {
        for (size_t j = 1; j < w - 1; ++j) {

            const size_t pos = i * w + j;
            assert (pos < w * h);

            //
            // A pixel is marked as foreground ... if it is presented(sic) in
            // the low-thresholded binarized mask AND it is spatially connected
            // to at least one pixel present in the high-thresholded binarized
            // mask:
            //
            if (p [pos] && (
                    q [pos] ||
                    q [pos - w - 1] ||
                    q [pos - w] ||
                    q [pos - w + 1] ||
                    q [pos - 1] ||
                    q [pos + 1] ||
                    q [pos + w - 1] ||
                    q [pos + w] ||
                    q [pos + w + 1])) {
                r [pos] = maxval;
            }
        }
    }

    return mask;
}

cv::Mat
temporal_median::threshold (
    const cv::Mat& src, const cv::Mat& mask, size_t maxval) {

    assert (src.size () == mask.size ());
    assert (src.type () == mask.type ());

    const unsigned char *p = src.data, *q = mask.data;

    cv::Mat result (src.size (), src.type ());
    unsigned char* r = result.data;

    for (size_t i = 0, n = src.total (); i < n; ++i)
        r [i] = (p [i] > q [i]) ? maxval : 0;

    return result;
}

const cv::Mat&
temporal_median::operator () (const cv::Mat& frame) {
    assert (frame.size ()     == background_.size ());
    assert (frame.channels () == background_.channels ());

    if (history_.size () < history_.capacity ()) {
        //
        // Store frames until the history buffer is full:
        //
        return history_.push_back (frame), mask_ = frame;
    }
    else {
        //
        // Compute the mask -- plain difference between the incoming frame
        // and background as (the formula is for RGB, we use gray levels):
        //
        // \begin{equation}
        // M(i,j) = \abs{I(i,j) - B(i,j)}
        // \end{equation}
        //
        cv::Mat diff = absdiff (frame, background_);

        //
        // Compute a "low threshold" mask and a "high threshold" mask:
        //
        // \begin{equation}
        // T_{lo}(i,j)= \lambda \times \left( b_{\frac{k+1}{2}+l} − b_{\frac{k+1}{2}-l} \right)
        // T_{hi}(i,j)= \lambda \times \left( b_{\frac{k+1}{2}+h} − b_{\frac{k+1}{2}-h} \right)
        // \end{equation}
        //
        // where l, h are the distances from the median position, and:
        //
        // \begin{equation}
        // l \leq h \leq \frac{history_size_}{2}
        // \end{equation}
        //
        // In the same pass update the background, which is the median of the
        // historic pixels:
        //
        cv::Mat lo_mask, hi_mask, background;
        std::tie (lo_mask, hi_mask, background) = calculate_masks ();

        if (0 == (++frame_counter_ % frame_interval_))
            history_.push_back (frame);

        lo_mask = threshold (diff, lo_mask);
        hi_mask = threshold (diff, hi_mask);

        mask_ = compose_masks (lo_mask, hi_mask, 255);

        background.copyTo (background_, 255 - mask_);

        return mask_;
    }
}

} // namespace bs
