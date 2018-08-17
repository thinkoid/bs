#include <bs/utils.hpp>
#include <bs/zivkovic_gmm.hpp>

#include <numeric>
using namespace std;

#include <opencv2/imgproc.hpp>

namespace bs {

inline zivkovic_gmm_t::gaussian_t
zivkovic_gmm_t::default_gaussian (const cv::Vec3b& arg) {
    return gaussian_t { variance_, 1., 1. / sqrt (variance_), arg };
}

/* explicit */
zivkovic_gmm_t::zivkovic_gmm_t (
    size_t n, double alpha, double variance_threshold, double variance,
    double weight_threshold, double bias)
    : size_ (n),
      alpha_ (alpha),
      variance_threshold_ (variance_threshold),
      variance_ (variance),
      weight_threshold_ (weight_threshold),
      bias_ (bias)
{ }

const cv::Mat&
zivkovic_gmm_t::operator() (const cv::Mat& frame) {
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
#pragma omp parallel for
        for (size_t i = 0; i < frame.total (); ++i) {
            const auto& src = frame.at< cv::Vec3b > (i);

            auto& gs = g_ [i];

            for_each (begin (gs), end (gs), [=](auto& g) {
                    g.s = g.w / sqrt (g.v); });

            sort (begin (gs), end (gs), [](const auto& g1, const auto& g2) {
                    return g1.s > g2.s; });

            size_t n = 0;

            for (double sum = 0.;
                 n < gs.size () && sum < weight_threshold_; ++n) {
                sum += gs [n].w;
            }

            int once = 0;

            for (size_t j = 0; j < gs.size (); ++j) {
                auto& g = gs [j];

                auto& v = g.v;
                auto& w = g.w;
                auto& m = g.m;

                const auto distance = dot (cv::Vec3d (src) - m);

                if (!once && distance < variance_threshold_ * v && ++once) {
                    if (j < n) {
                        //
                        // If the distance is close enough to a distribution
                        // that models the background:
                        //
                        mask_.at< unsigned char > (i) = 0;
                        background_.at< cv::Vec3b > (i) = gs [0].m;
                    }

                    const double r = alpha_ * w - alpha_ * bias_;

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
                    w = (1. - alpha_) * w - alpha_ * bias_;
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

            {
                //
                // Sort by weights, prune:
                //
                sort (begin (gs), end (gs), [](const auto& g1, const auto& g2) {
                        return g1.w > g2.w; });

                auto last = find_if (begin (gs), end (gs), [=](auto& g) {
                        return g.w < 0.; });

                gs.erase (last, end (gs));
            }

            {
                //
                // Re-normalize weights:
                //
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
