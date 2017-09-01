#include <bs/utils.hpp>
#include <bs/grimson_gmm.hpp>

#include <numeric>
using namespace std;

#include <opencv2/imgproc.hpp>

namespace {

inline double
euclidean_distance (const cv::Vec3b& x, const cv::Vec3b& y) {
    const double a = x [0] - y [0], b = x [1] - y [1], c = x [2] - y [2];
    return a * a + b * b + c * c;
}

}

namespace bs {

inline grimson_gmm_t::gaussian_t
grimson_gmm_t::default_gaussian (const cv::Vec3b& arg) {
    return gaussian_t { variance_, 1., 1. / sqrt (variance_), arg };
}

/* explicit */
grimson_gmm_t::grimson_gmm_t (
    size_t n, double alpha, double variance_threshold, double variance,
    double background_threshold)
    : size_ (n),
      alpha_ (alpha),
      variance_threshold_ (variance_threshold),
      variance_ (variance),
      background_threshold_ (background_threshold)
{ }

const cv::Mat&
grimson_gmm_t::operator() (const cv::Mat& frame) {
    mask_ = cv::Mat (frame.size (), CV_8U, cv::Scalar (255));

    if (g_.empty ()) {
        g_.resize (frame.total ());

        for (size_t i = 0; i < g_.size (); ++i) {
            auto& g = g_ [i];
            g.reserve (size_);
            g.resize (1UL, default_gaussian (frame.at< cv::Vec3b > (i)));
        }

        background_ = frame.clone ();
    }
    else {
        for (size_t i = 0; i < frame.total (); ++i) {
            const auto& src = frame.at< cv::Vec3b > (i);

            auto& gs = g_ [i];

            for_each (begin (gs), end (gs), [=](auto& g) {
                    g.s = g.w / sqrt (g.v); });

            sort (begin (gs), end (gs), [](const auto& g1, const auto& g2) {
                    return g1.s > g2.s; });

            size_t n = 0;

            for (double sum = 0.;
                 n < gs.size () && sum < background_threshold_; ++n) {
                sum += gs [n].w;
            }

            int once = 0;

            for (size_t j = 0; j < gs.size (); ++j) {
                auto& g = gs [j];

                auto& v = g.v;
                auto& w = g.w;
                auto& m = g.m;

                const auto distance = euclidean_distance (src, m);

                if (!once && distance < variance_threshold_ * v && ++once) {
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
                    gs.emplace_back (gaussian_t {
                            variance_, alpha_, alpha_ / sqrt (variance_), src });
                }
                else {
                    gs.back () = gaussian_t {
                        variance_, alpha_, alpha_ / sqrt (variance_), src };
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
