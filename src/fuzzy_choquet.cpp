#include <bs/fuzzy_choquet.hpp>
#include <bs/detail/lbp.hpp>
#include <bs/utils.hpp>

#include <iostream>
#include <algorithm>
#include <numeric>
using namespace std;

#include <opencv2/imgproc.hpp>
using namespace cv;

#include "fuzzy_integral.hpp"

namespace bs {

/* explicit */
fuzzy_choquet_t::fuzzy_choquet_t (
    const cv::Mat& b, double a, double t, const vector< double >& g)
    : detail::base_t (b), alpha_ (a), threshold_ (t), g_ (g)
{ }

const cv::Mat&
fuzzy_choquet_t::operator() (const cv::Mat& frame) {
    BS_ASSERT (3 == frame.channels ());

    Mat fframe = convert (
        convert_color (frame, cv::COLOR_BGR2YCrCb), CV_32F, 1./255);

    const auto F = lbp (gray_from (fframe)) / 255.;
    const auto B = lbp (gray_from (background_)) / 255.;

    const auto H = similarity1 (F, B);
    const auto I = similarity3 (fframe, background_);

    const auto S = median_blur (choquet_integral (H, I, g_));

    background_ = update_background (fframe, background_, S, alpha_);

    return mask_ = convert (
        threshold (S, threshold_, 255.f, THRESH_BINARY_INV), CV_8U, 255.f);
}

}
