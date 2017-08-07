#include <bs/fuzzy_choquet.hpp>
#include <bs/detail/lbp.hpp>
#include <bs/utils.hpp>

#include <iostream>
#include <algorithm>
#include <numeric>
using namespace std;

#include <opencv2/imgproc.hpp>
using namespace cv;

namespace {

inline float
h_texture (float lhs, float rhs)
{
    return lhs < rhs ? lhs / rhs : lhs > rhs ? rhs / lhs : 1.0;
}

inline Vec3f
h_texture (const Vec3f& lhs, const Vec3f& rhs)
{
    Vec3f result;

#define T(a, b, c) c = a < b ? a/b : a > b ? b/a : 1.f;

    T (lhs [0], rhs [0], result [0]);
    T (lhs [1], rhs [1], result [1]);
    T (lhs [2], rhs [2], result [2]);

#undef T

    return result;
}

Mat
similarity1 (const Mat& fg, const Mat& bg)
{
    Mat d = Mat (fg.size (), CV_32F, Scalar (0));

    for (size_t i = 0; i < fg.total (); ++i) {
        d.at< float > (i) = h_texture (fg.at< float > (i), bg.at< float > (i));
    }

    return d;
}

Mat
similarity3 (const Mat& fg, const Mat& bg)
{
    Mat d = Mat (fg.size (), CV_32FC3, Scalar (0));

    for (size_t i = 0; i < fg.total (); ++i) {
        d.at< Vec3f > (i) = h_texture (fg.at< Vec3f > (i), bg.at< Vec3f > (i));
    }

    return d;
}

Mat
choquet_integral (const Mat& H, const Mat& I, const vector< double >& g)
{
    Mat S (H.size (), CV_32F);

    for (size_t i = 0; i < H.total (); ++i) {
        const auto h = H.at< float > (i);
        const auto d = I.at< Vec3f > (i);

        S.at< float > (i) = h * g [0] + d [0] * g [1] + d [1] * g [2];
    }

    return S;
}

Mat
update_background (const Mat& F, const Mat& B, const Mat& S, float alpha)
{
    Mat result (F.size (), CV_32FC3, Scalar (0));

    double max_, min_;
    std::tie (min_, max_) = bs::minmax (S);

    BS_ASSERT (min_ >= 0. && max_ >= 0. && 1.0 >= min_ && 1.0 >= max_);

    for (size_t i = 0; i < F.total (); ++i) {
        auto& dst = result.at< Vec3f > (i);

        const auto& f = F.at< Vec3f > (i);
        const auto& b = B.at< Vec3f > (i);
        const auto& s = S.at< float > (i);

        const auto beta = 1. - max_ * (s - min_) / (max_ - min_);

        dst [0] = beta * b [0] + (1 - beta) * (alpha * f [0] + (1 - alpha) * b [0]);
        dst [1] = beta * b [1] + (1 - beta) * (alpha * f [1] + (1 - alpha) * b [1]);
        dst [2] = beta * b [2] + (1 - beta) * (alpha * f [2] + (1 - alpha) * b [2]);
    }

    return result;
}

}

namespace bs {

/* explicit */
fuzzy_choquet_bootstrap_t::fuzzy_choquet_bootstrap_t (
    double alpha, size_t frame_counter)
    : alpha_ (alpha), frame_counter_ (frame_counter)
{ }

bool
fuzzy_choquet_bootstrap_t::process (const cv::Mat& frame) {
    Mat fframe (frame.size (), CV_32F);
    frame.convertTo (fframe, CV_32F, 1. / 255);

    if (background_.empty ()) {
        background_ = fframe;
    }
    else {
        background_ = alpha_ * fframe + (1 - alpha_) * background_;
    }

    return 0 == --frame_counter_;
}

/* explicit */
fuzzy_choquet_t::fuzzy_choquet_t (
    const cv::Mat& b, double a, double t, const vector< double >& g)
    : detail::base_t (b), alpha_ (a), threshold_ (t), g_ (g)
{ }

const cv::Mat&
fuzzy_choquet_t::operator() (const cv::Mat& frame) {
    BS_ASSERT (3 == frame.channels ());

    Mat fframe = convert (convert_color (frame, CV_BGR2YCrCb), CV_32F, 1./255);

    const auto F = detail::lbp (gray_from (fframe)) / 255.;
    const auto B = detail::lbp (gray_from (background_)) / 255.;

    const auto H = similarity1 (F, B);
    const auto I = similarity3 (fframe, background_);

    const auto S = median_blur (choquet_integral (H, I, g_));

    background_ = update_background (fframe, background_, S, alpha_);

    return mask_ = convert (
        threshold (S, threshold_, 255.f, THRESH_BINARY_INV), CV_8U, 255.f);
}

}
