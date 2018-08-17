#ifndef BS_FUZZY_INTEGRAL_HPP
#define BS_FUZZY_INTEGRAL_HPP

#include <bs/defs.hpp>

#include <opencv2/imgproc.hpp>
using namespace cv;

namespace {

inline void
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

inline double
h_texture (double lhs, double rhs, double off = 1./255)
{
    return (lhs + off) < rhs ? lhs / rhs : lhs > (rhs + off) ? rhs / lhs : 1;
}

inline Vec3f
h_texture (const Vec3f& lhs, const Vec3f& rhs, double off = 1./255)
{
    Vec3f result;

#define T(a, b, c) c = h_texture (a, b, off)

    T (lhs [0], rhs [0], result [0]);
    T (lhs [1], rhs [1], result [1]);
    T (lhs [2], rhs [2], result [2]);

#undef T

    return result;
}

inline Mat
similarity1 (const Mat& fg, const Mat& bg)
{
    Mat d = Mat (fg.size (), CV_32F, Scalar (0));

#pragma omp parallel for
    for (size_t i = 0; i < fg.total (); ++i) {
        d.at< float > (i) = h_texture (fg.at< float > (i), bg.at< float > (i));
    }

    return d;
}

inline Mat
similarity3 (const Mat& fg, const Mat& bg)
{
    Mat d = Mat (fg.size (), CV_32FC3, Scalar (0));

    for (size_t i = 0; i < fg.total (); ++i) {
        d.at< Vec3f > (i) = h_texture (fg.at< Vec3f > (i), bg.at< Vec3f > (i));
    }

    return d;
}

inline Mat
choquet_integral (const Mat& H, const Mat& I, const vector< double >& g)
{
    Mat S (H.size (), CV_32F);

#pragma omp parallel for
    for (size_t i = 0; i < H.total (); ++i) {
        const auto h = H.at< float > (i);
        const auto d = I.at< Vec3f > (i);

        S.at< float > (i) = h * g [0] + d [0] * g [1] + d [1] * g [2];
    }

    return S;
}

inline Mat
sugeno_integral (const Mat& H, const Mat& I, const vector< double >& g)
{
    double h_x [3], s [3];

    Mat S (H.size (), CV_32F);

#pragma omp parallel for
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

inline Mat
update_background (const Mat& F, const Mat& B, const Mat& S, float alpha)
{
    Mat result (F.size (), CV_32FC3, Scalar (0));

    double min_, max_;
    std::tie (min_, max_) = bs::minmax (S);

#pragma omp parallel for
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

#endif // BS_FUZZY_INTEGRAL_HPP
