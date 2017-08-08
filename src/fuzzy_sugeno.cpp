#include <bs/fuzzy_sugeno.hpp>
#include <bs/detail/lbp.hpp>
#include <bs/utils.hpp>

#include <algorithm>
using namespace std;

#include <opencv2/imgproc.hpp>
using namespace cv;

#include "fuzzy_integral.hpp"

namespace bs {

/* explicit */
fuzzy_sugeno_t::fuzzy_sugeno_t (
    const cv::Mat& b, double a, double t, const vector< double >& g)
    : detail::base_t (b), alpha_ (a), threshold_ (t), g_ (g)
{ }

const cv::Mat&
fuzzy_sugeno_t::operator() (const cv::Mat& frame) {
    BS_ASSERT (3 == frame.channels ());

    Mat fframe = convert (convert_ohta (frame), CV_32F, 1./255);

    const auto F = lbp (gray_from (fframe)) / 255.;
    const auto B = lbp (gray_from (background_)) / 255.;

    const auto H = similarity1 (F, B);
    const auto I = similarity3 (fframe, background_);

    //
    // Note: for well-chosen densities whose sum is 1.0, the parameter λ
    // in the Sugeno λ-measure becomes 0, thus simplifying the subsequent
    // calculations:
    //
    const auto S = median_blur (sugeno_integral (H, I, g_));

    background_ = update_background (fframe, background_, S, alpha_);

    return mask_ = convert (
        threshold (S, threshold_, 255.f, THRESH_BINARY_INV), CV_8U, 255.f);
}

}
