#include <bs/utils.hpp>
#include <bs/sigma_delta.hpp>

#include <opencv2/imgproc.hpp>

namespace bs {

/* explicit */
sigma_delta::sigma_delta (
    const cv::Mat& background, int n, int threshold)
    : mask_ (background.size (), CV_8U, cv::Scalar (0)),
      m_ (background.clone ()),
      d_ (background.size (), CV_8U, cv::Scalar (0)),
      v_ (background.size (), CV_8U, cv::Scalar (0)),
      n_ (n), threshold_ (threshold) {
}

const cv::Mat&
sigma_delta::operator() (const cv::Mat& frame) {
    //
    // m_ is M_t, a running approximation of the median:
    //
    m_ = m_ + threshold (frame - m_, 0, 1) - threshold (m_ - frame, 0, 1);

    //
    // d_ is Δ_t, an absolute difference between the frame and the
    // running median:
    //
    d_ = absdiff (frame, m_);

    //
    // ... we also use this filter to compute the time-variance of the pixels,
    // representing their motion activity measure, used to decide whether the
    // pixel is more likely "moving" or "stationary".
    //
    // Then, v_ (V_t) ... used in our method has the dimension of a temporal
    // standard deviation. It is computed as a Σ-Δ filter of the difference
    // sequence d_ (D_t). This provides a measure of temporal activity of the
    // pixels. As we are interested in pixels whose variation rate is
    // significantly over its temporal activity, we apply the Σ-Δ filter to the
    // sequence of N times the non-zero differences.
    //
    // Finally, the pixel-level detection is simply performed by comparing d_
    // (D_t) and v_ (V_t) (Table 1(4)).
    //

    cv::Mat tmp = n_ * d_;
    v_ = v_ + threshold (tmp - v_, 1, 1) - threshold (v_ - tmp, 1, 1);

    return mask_ = threshold (d_ - v_, threshold_, 255);
}

} // namespace bs
