#include <bs/fuzzy_sugeno.hpp>
#include <bs/detail/lbp.hpp>
#include <bs/utils.hpp>

#include <algorithm>
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

//
// On similarity:
//
// [...] uncertainty h_{texture} which is deﬁned as [...], where
// G_B(x_c,y_c) is the texture LBP of pixel (x_c,y_c) in background and
// G_t(x_c,y_c) is the texture LBP of pixel (x_c,y_c) in time t video
// frame. similarity is close to one if G_B(x_c,y_c) and
// G_t(x_c,y_c) are very similar.
//
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

void
sort3 (double* f, int* i)
{
    static const int arr [][3] = {
        { 2, 1, 0 },
        { 1, 2, 0 },
        { 0, 0, 0 },
        { 1, 0, 2 },
        { 2, 0, 1 },
        { 0, 0, 0 },
        { 0, 2, 1 },
        { 0, 1, 2 }
    };

    const auto& a = f [0];
    const auto& b = f [1];
    const auto& c = f [2];

    unsigned mask =
        (a >= b ? 4 : 0) |
        (a >= c ? 2 : 0) |
        (b >= c ? 1 : 0);

    const auto& p = arr [mask];

    i [0] = p [0];
    i [1] = p [1];
    i [2] = p [2];
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
        sort3 (h_x, index);

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
fuzzy_sugeno_bootstrap_t::fuzzy_sugeno_bootstrap_t (
    double alpha, size_t frame_counter)
    : alpha_ (alpha), frame_counter_ (frame_counter)
{ }

bool
fuzzy_sugeno_bootstrap_t::process (const cv::Mat& frame) {
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
fuzzy_sugeno_t::fuzzy_sugeno_t (
    const cv::Mat& b, double a, double t, const vector< double >& g)
    : detail::base_t (b), alpha_ (a), threshold_ (t), g_ (g)
{ }

const cv::Mat&
fuzzy_sugeno_t::operator() (const cv::Mat& frame) {
    BS_ASSERT (3 == frame.channels ());

    Mat fframe = convert (frame, CV_32F, 1./255);

    const auto F = detail::lbp (gray_from (fframe)) / 255.;
    const auto B = detail::lbp (gray_from (background_)) / 255.;

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
