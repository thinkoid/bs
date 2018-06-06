#include <bs/utils.hpp>
#include <bs/sigma_delta.hpp>

#include <opencv2/imgproc.hpp>

#include <iostream>
using namespace std;

namespace bs {

/* explicit */
sigma_delta_t::sigma_delta_t (
    const cv::Mat& b, size_t n, size_t Vmin, size_t Vmax)
    : detail::base_t (b, { b.size (), CV_8U, cv::Scalar (0) }),
      m_ (b.clone ()),
      d_ (b.size (), CV_8U, cv::Scalar (0)),
      v_ (b.size (), CV_8U, cv::Scalar (0)),
      q_ (b.size (), CV_8U, cv::Scalar (255)),
      n_ (n), Vmin_ (Vmin), Vmax_ (Vmax)
{ }

const cv::Mat&
sigma_delta_t::operator() (const cv::Mat& frame) {
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
    // (D_t) and v_ (V_t)...
    //

    //
    // sgn(N×Δ_t - V_{t-1})
    //
    cv::Mat tmp = n_ * d_;
    tmp = threshold (tmp - v_, 0, 1) - threshold (v_ - tmp, 0, 1);

    //
    // Mask all Δ_t zeroes:
    //
    cv::Mat mask = threshold (d_, 1, 1);
    tmp = tmp.mul (mask);

    //
    // V_t = V_{t-1} + sgn(N×Δ_t - V_{t-1}), Δ_t ≠ 0
    //
    v_ = v_ + tmp;

    //
    // V_t = max(min(Vmax, V_t), Vmin)
    //
    v_ = threshold (v_, Vmax_, 0, cv::THRESH_TRUNC);

    v_ = q_ - v_;
    v_ = threshold (v_, 255 - Vmin_, 0, cv::THRESH_TRUNC);

    v_ = q_ - v_;

    //
    // Ê_t = (O_t < V_t) ? 0 : 1
    //
    return mask_ = threshold (d_ - v_, 0, 255);
}

}
