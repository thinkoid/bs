#include <bs/utils.hpp>
#include <bs/fgmm.hpp>

namespace bs {

template< typename T >
const cv::Mat&
fgmm_base_t< T >::operator() (const cv::Mat& frame) {
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

                const double distance = sqrt (
                    dot (f_ (cv::Vec3d (src), m, v, s, k_)));

                if (0 == once && distance < variance_threshold_ * s &&
                    1 == ++once) {

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

} // namespace bs
