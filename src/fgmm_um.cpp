#include <bs/utils.hpp>
#include <bs/fgmm_um.hpp>

#include <numeric>
using namespace std;

#include <opencv2/imgproc.hpp>

namespace {

static inline double
dot (const cv::Vec3d& x, const cv::Vec3d& y) {
    return x [0] * y [0] + x [1] * y [1] + x [2] * y [2];
}

static inline double
dot (const cv::Vec3d& x) {
    return dot (x, x);
}

static inline double
clamp (double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi: x);
}

//
// Gaussian primary membership function with uncertain mean:
//
static inline cv::Vec3d
mfum (const cv::Vec3d& x, const cv::Vec3d& y, double v, double s, double k) {
    BS_ASSERT (v > 0);
    BS_ASSERT (s > 0);
    BS_ASSERT (k > 0);

    const cv::Vec3d d = y - x;

    cv::Vec3d z;

    z [0] = (x [0] < y [0] - k * s) || (x [0] > y [0] + k * s)
        ? 2 * k * d [0] / s
        : d [0] / (2 * v) + k * d [0] / s + k * k / 2;

    z [1] = (x [1] < y [1] - k * s) || (x [1] > y [1] + k * s)
        ? 2 * k * d [1] / s
        : d [1] / (2 * v) + k * d [1] / s + k * k / 2;

    z [2] = (x [2] < y [2] - k * s) || (x [2] > y [2] + k * s)
        ? 2 * k * d [2] / s
        : d [2] / (2 * v) + k * d [2] / v + k * k / 2;

    return z;
}

}

namespace bs {

inline fgmm_um_t::gaussian_t
fgmm_um_t::make_gaussian (const cv::Vec3b& src, double v, double s, double a) {
    return gaussian_t { v, s, a, a / s, cv::Vec3d (src) };
}

inline fgmm_um_t::gaussian_t
fgmm_um_t::make_gaussian (const cv::Vec3b& src, double v, double a) {
    return make_gaussian (src, v, sqrt (v), a);
}

inline fgmm_um_t::gaussian_t
fgmm_um_t::make_gaussian (const cv::Vec3b& src, double v) {
    return make_gaussian (src, v, sqrt (v), 1.);
}

/* explicit */
fgmm_um_t::fgmm_um_t (
    size_t n,
    double alpha,
    double variance,
    double variance_threshold,
    double weight_threshold,
    double k)
    : size_ (n),
      alpha_ (alpha),
      variance_ (variance),
      variance_threshold_ (variance_threshold),
      weight_threshold_ (weight_threshold),
      k_ (k)
{ }

const cv::Mat&
fgmm_um_t::operator() (const cv::Mat& frame) {
    mask_ = cv::Mat (frame.size (), CV_8U, cv::Scalar (255));

    if (g_.empty ()) {
        g_.resize (frame.total ());

        for (size_t i = 0; i < g_.size (); ++i) {
            auto& g = g_ [i];
            g.reserve (size_);

            const auto& src = frame.at< cv::Vec3b > (i);
            g.resize (1UL, make_gaussian (src, variance_));
        }

        background_ = frame.clone ();
    }
    else {
#pragma omp parallel for
        for (size_t i = 0; i < frame.total (); ++i) {
            const auto& src = frame.at< cv::Vec3b > (i);

            auto& gs = g_ [i];

            for_each (begin (gs), end (gs), [=](auto& g) {
                    g.g = g.w / g.s; });

            sort (begin (gs), end (gs), [](const auto& g1, const auto& g2) {
                    return g1.g > g2.g; });

            size_t n = 0;

            for (double sum = 0.;
                 n < gs.size () && sum < weight_threshold_; ++n) {
                sum += gs [n].w;
            }

            int once = 0;

            for (size_t j = 0; j < gs.size (); ++j) {
                auto& g = gs [j];

                auto& v = g.v;
                auto& s = g.s;
                auto& w = g.w;
                auto& m = g.m;

                const double ll = dot (mfum (cv::Vec3d (src), m, v, s, k_));

                if (0 == once && ll < variance_threshold_ * v && 1 == ++once) {
                    if (j < n) {
                        mask_.at< unsigned char > (i) = 0;
                        background_.at< cv::Vec3b > (i) = gs [0].m;
                    }

                    const double r = alpha_ * w;

                    w = (1. - alpha_) * w + alpha_;

                    m [0] += r * (src [0] - m [0]);
                    m [1] += r * (src [1] - m [1]);
                    m [2] += r * (src [2] - m [2]);

                    v += r * (dot (cv::Vec3d (src) - cv::Vec3d (m)) - v);
                    s = sqrt (v);
                }
                else {
                    w = (1. - alpha_) * w;
                }
            }

            if (!once) {
                const auto& src = frame.at< cv::Vec3b > (i);

                if (gs.size () < size_) {
                    gs.emplace_back (make_gaussian (src, variance_, alpha_));
                }
                else {
                    gs.back () = make_gaussian (src, variance_, alpha_);
                }
            }

            {
                sort (begin (gs), end (gs), [](const auto& g1, const auto& g2) {
                        return g1.w > g2.w; });

                auto last = find_if (begin (gs), end (gs), [=](auto& g) {
                    return g.w < 0; });

                gs.erase (last, end (gs));
            }

            {
                const auto normal = 1. / accumulate (
                    begin (gs), end (gs), 0., [](auto accum, const auto& g) {
                        return accum + g.w; });

                for_each (begin (gs), end (gs), [=](auto& g) { g.w *= normal; });
            }
        }
    }

    return mask_;
}

}
