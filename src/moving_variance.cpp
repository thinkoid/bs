#include <bs/utils.hpp>
#include <bs/moving_variance.hpp>

#include <opencv2/imgproc.hpp>

namespace bs {
namespace detail {

inline cv::Mat
weigh (const cv::Mat& arg, const double weight) {
    return weight * arg;
}

inline cv::Mat
weighted_variance_of (
    const cv::Mat& input, const cv::Mat& mean, const double weight) {
    return weigh (square_of (absdiff (input, mean)), weight);
}

} // namespace detail

////////////////////////////////////////////////////////////////////////

/* explicit */
moving_variance::moving_variance (
    std::vector< double > weights, int threshold)
    : framebuf_ (3), weights_ (weights), threshold_ (threshold)
{ }

const cv::Mat&
moving_variance::operator() (const cv::Mat& frame) {
    framebuf_.push_back (float_from (frame));

    if (framebuf_.size () < 3)
        return mask_ = mono_from (frame);

    cv::Mat mean =
        framebuf_ [0] * weights_ [0] +
        framebuf_ [1] * weights_ [0] +
        framebuf_ [2] * weights_ [2];

    cv::Mat s =
        detail::weighted_variance_of (framebuf_ [0], mean, weights_ [0]) +
        detail::weighted_variance_of (framebuf_ [1], mean, weights_ [1]) +
        detail::weighted_variance_of (framebuf_ [2], mean, weights_ [2]);

    cv::sqrt (s, s);

    return mask_ = threshold (mono_from (s), threshold_);
}

}
