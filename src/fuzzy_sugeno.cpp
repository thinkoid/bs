#include <bs/utils.hpp>
#include <bs/fuzzy_sugeno.hpp>

#include <algorithm>
using namespace std;

#include <opencv2/imgproc.hpp>
using namespace cv;

namespace {

Mat
LBP (const Mat& src)
{
    //
    // Ojala, 2001 - A Generalized Local Binary Pattern Operator ...
    // Note: does not compute the uniformity measure U.
    //
    auto dst = Mat::zeros (src.size (), CV_32F);

    for (size_t i = 1; i < size_t (src.rows) - 1; i++) {
        const float* p = src.ptr< float > (i - 1);
        const float* q = src.ptr< float > (i);
        const float* r = src.ptr< float > (i + 1);

        float* s = dst.ptr< float > (i);
        ++s;

        for (size_t j = 1; j < size_t (src.cols) - 1; ++j, ++p, ++q, ++r, ++s) {
            float t = q [1];

            const auto tmp =
                (unsigned (p [0] >= t) << 7) +
                (unsigned (p [1] >= t) << 6) +
                (unsigned (p [2] >= t) << 5) +
                (unsigned (q [0] >= t)) +
                (unsigned (q [2] >= t) << 4) +
                (unsigned (r [0] >= t) << 1) +
                (unsigned (r [1] >= t) << 2) +
                (unsigned (r [2] >= t) << 3);

            assert (tmp >= 0.);
            s [0] = tmp / 255.;
        }
    }

    return dst;
}

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
similarity_of (const Mat& fg, const Mat& bg)
{
    //
    // [...] uncertainty h_{texture} which is deﬁned as [...], where
    // G_B(x_c,y_c) is the texture LBP of pixel (x_c,y_c) in background and
    // G_t(x_c,y_c) is the texture LBP of pixel (x_c,y_c) in time t video
    // frame. similarity is close to one if G_B(x_c,y_c) and
    // G_t(x_c,y_c) are very similar.
    //

    switch (fg.channels ()) {
    case 1: return similarity1 (fg, bg);
    case 3: return similarity3 (fg, bg);
    default:
        assert (0);
    }
}

Mat
sugeno_integral (const Mat& H, const Mat& I, const vector< double >& g)
{
    double h_x [3], s [3];

    Mat S (H.size (), CV_32F);

    for (size_t i = 0; i < H.total (); ++i) {
        const auto h = H.at< float > (i);
        const auto d = I.at< Vec3f > (i);

        //
        // Certainly, the feature sets is X = {x_1, x_2, x_3 }. One element is
        // x_1 = {texture} and the others are x_2 = { I_1 } and x_3 = { I_2 }
        // [...] Let h_i : X → [0,1] be a fuzzy function. Fuzzy function
        // h_1 = h(x_1) = h_{texture} is the evaluation of texture feature.
        // Fuzzy function h_2 = h(x_2) = h_{ΔI_1} is the evaluation of color
        // feature I_1. Fuzzy function h_3 = h(x_3) = h_{ΔI_2} is the
        // evaluation of color feature I_2.
        //
        h_x [0] = h;
        h_x [1] = d [0];
        h_x [2] = d [1];

        int index [3] = { 0, 1, 2 };

        //
        // The calculation of the fuzzy integral is as follows: suppose
        // h(x_1) ≥ h(x_2) ≥ h(x_3), if not, X is rearranged so that
        // this relation holds [...]
        //
#define T(x, y)                                 \
        if (h_x [x] < h_x [y]) {                \
            swap (h_x [x], h_x [y]);            \
            swap (index [x], index [y]);        \
        }

        T (1, 2);
        T (0, 1);
        T (1, 2);
#undef T

        h_x [0] = h;
        h_x [1] = d [0];
        h_x [2] = d [1];

        //
        // [...] A fuzzy integral, S, with respect to a fuzzy measure g
        // over X can be computed by S = max_{i=1}^n[min(h(x_i), g(X_i))]:
        //
        s [0] = (min) (h_x [index [0]], 1.);
        s [1] = (min) (h_x [index [1]], g [index [1]] + g [index [2]]);
        s [2] = (min) (h_x [index [2]], g [index [2]]);

        S.at< float > (i) = max_element (s, s + 3)[0];
    }

    return S;
}

Mat
update_background (const Mat& fg, const Mat& bg, const Mat& S, float alpha)
{
    Mat result (fg.size (), CV_32FC3, Scalar (0));

    double max_, min_;
    minMaxLoc (S, &min_, &max_);

    assert (min_ >= 0. && max_ >= 0. && 1.0 >= min_ && 1.0 >= max_);

    for (size_t i = 0; i < fg.total (); ++i) {
        const auto& f = fg.at< Vec3f > (i);
        const auto& b = bg.at< Vec3f > (i);

        const auto& s = S.at< float > (i);

        auto& dst = result.at< Vec3f > (i);

        const auto beta = 1. - (s - min_ * ((max_ - s) / (max_ - min_)));

        dst [0] = beta * b [0] + (1 - beta) * (alpha * f [0] + (1 - alpha) * b [0]);
        dst [1] = beta * b [1] + (1 - beta) * (alpha * f [1] + (1 - alpha) * b [1]);
        dst [2] = beta * b [2] + (1 - beta) * (alpha * f [2] + (1 - alpha) * b [2]);
    }

    return result;
}

} // anonymous

namespace bs {

/* explicit */
fuzzy_sugeno_bootstrap::fuzzy_sugeno_bootstrap (
    double alpha, size_t frame_counter)
    : alpha_ (alpha), frame_counter_ (frame_counter)
{ }

bool
fuzzy_sugeno_bootstrap::process (const cv::Mat& frame) {
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
fuzzy_sugeno::fuzzy_sugeno (
    const cv::Mat& background,
    const vector< double >& g, double alpha, double threshold)
    : background_ (background),
      g_ (g), alpha_ (alpha), threshold_ (threshold)
{ }

const cv::Mat&
fuzzy_sugeno::operator() (const cv::Mat& frame) {
    assert (3 == frame.channels ());

    Mat fframe (frame.size (), CV_32F);
    frame.convertTo (fframe, CV_32F, 1. / 255);

    const auto lbp_fg = LBP (gray_from (fframe));
    const auto lbp_bg = LBP (gray_from (background_));

    const auto H = similarity_of (lbp_fg, lbp_bg);
    const auto I = similarity_of (fframe, background_);

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
