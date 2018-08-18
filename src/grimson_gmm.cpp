#include <bs/utils.hpp>
#include <bs/grimson_gmm.hpp>

#include <numeric>
using namespace std;

#include <opencv2/imgproc.hpp>

namespace bs {

inline grimson_gmm_t::gaussian_t
grimson_gmm_t::make_gaussian (const cv::Vec3b& src, double v, double s, double a) {
    return gaussian_t { v, s, a, a / s, cv::Vec3d (src) };
}

inline grimson_gmm_t::gaussian_t
grimson_gmm_t::make_gaussian (const cv::Vec3b& src, double v, double a) {
    return make_gaussian (src, v, sqrt (v), a);
}

inline grimson_gmm_t::gaussian_t
grimson_gmm_t::make_gaussian (const cv::Vec3b& src, double v) {
    return make_gaussian (src, v, sqrt (v), 1.);
}

/* explicit */
grimson_gmm_t::grimson_gmm_t (
    size_t n, double alpha, double variance_threshold, double variance,
    double weight_threshold)
    : size_ (n),
      alpha_ (alpha),
      variance_threshold_ (variance_threshold),
      variance_ (variance),
      weight_threshold_ (weight_threshold)
{ }

const cv::Mat&
grimson_gmm_t::operator() (const cv::Mat& frame) {
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

                const auto distance = sqrt (dot (cv::Vec3d (src) - m));

                if (!once && distance < variance_threshold_ * s && ++once) {
                    if (j < n) {
                        //
                        // If the distance is close enough to a distribution
                        // that models the background:
                        //
                        mask_.at< unsigned char > (i) = 0;
                        background_.at< cv::Vec3b > (i) = gs [0].m;
                    }

                    const double r = alpha_ * w;

                    w = (1. - alpha_) * w + alpha_;

                    m [0] += r * (src [0] - m [0]);
                    m [1] += r * (src [1] - m [1]);
                    m [2] += r * (src [2] - m [2]);

                    v += r * (distance - v);
                    s = sqrt (v);
                }
                else {
                    //
                    // All other distributions are unchanged:
                    //
                    w = (1. - alpha_) * w;
                }
            }

            if (!once) {
                //
                // No matching will create a new distribution or replace the
                // weakest (least probable):
                //
                if (gs.size () < size_) {
                    gs.emplace_back (make_gaussian (src, variance_, alpha_));
                }
                else {
                    gs.emplace_back (make_gaussian (src, variance_, alpha_));
                }
            }

            //
            // Re-normalize the weights:
            //
            const auto normal = 1. / accumulate (
                begin (gs), end (gs), 0., [](auto accum, const auto& g) {
                    return accum + g.w;
                });

            for_each (begin (gs), end (gs), [=](auto& g) { g.w *= normal; });
        }
    }

    return mask_;
}

}
