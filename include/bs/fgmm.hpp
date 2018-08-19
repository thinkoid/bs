#ifndef BS_FGMM_HPP
#define BS_FGMM_HPP

#include <bs/defs.hpp>
#include <bs/detail/base.hpp>

#include <numeric>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace bs {

//
// @inproceedings{el2008type,
//   title={Type-2 fuzzy mixture of Gaussians model: application to background
//   modeling}, author={El Baf, Fida and Bouwmans, Thierry and Vachon,
//   Bertrand}, booktitle={International Symposium on Visual Computing},
//   pages={772--781}, year={2008}, organization={Springer}
// }
//

//
// Gaussian primary membership function with uncertain mean:
//
struct mfum_t {
    cv::Vec3d
    operator() (const cv::Vec3d& x, const cv::Vec3d& y,
               double v, double s, double k) {
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
};

struct mfuv_t {
    cv::Vec3d
    operator() (const cv::Vec3d& x, const cv::Vec3d& y,
                double v, double s, double k) {
        BS_ASSERT (v > 0);
        BS_ASSERT (s > 0);
        BS_ASSERT (k > 0);

        const cv::Vec3d d = y - x;

        return ((1 / (k * k) - k * k) / (2 * v)) * d.mul (d);
    }
};

template< typename F >
struct fgmm_base_t : detail::base_t {
    static constexpr auto default_modes = 4.;
    static constexpr auto default_alpha = .005;
    static constexpr auto default_variance = 16.;
    static constexpr auto default_variance_threshold = 2.5;
    static constexpr auto default_weight_threshold = .7;

public:
    explicit fgmm_base_t (
        size_t n, double a, double v, double t, double w, double k, const F& f)
        : size_ (n), alpha_ (a), variance_ (v), variance_threshold_ (t),
          weight_threshold_ (w), k_ (k), f_ (f)
        { }

public:
    const cv::Mat&
    operator() (const cv::Mat&);

private:
    struct gaussian_t {
        double v, s, w, g;
        cv::Vec3d m;
    };

    gaussian_t
    make_gaussian (const cv::Vec3b& src, double v, double s, double a) const {
        return gaussian_t { v, s, a, a / s, cv::Vec3d (src) };
    }

    gaussian_t
    make_gaussian (const cv::Vec3b& src, double v, double a) const {
        return make_gaussian (src, v, sqrt (v), a);
    }

    gaussian_t
    make_gaussian (const cv::Vec3b& src, double v) const {
        return make_gaussian (src, v, sqrt (v), 1.);
    }


private:
    size_t size_;
    double alpha_, variance_, variance_threshold_,weight_threshold_, k_;
    std::vector< std::vector< gaussian_t > > g_;

    F f_;
};

struct fgmm_um_t : fgmm_base_t< mfum_t > {
    using base_type = fgmm_base_t< mfum_t >;

public:
    static constexpr auto default_k = 2.5;

public:
    explicit fgmm_um_t (
        size_t n = base_type::default_modes,
        double a = base_type::default_alpha,
        double v = base_type::default_variance,
        double t = base_type::default_variance_threshold,
        double w = base_type::default_weight_threshold,
        double k = default_k)
        : base_type (n, a, v, t, w, k, mfum_t ())
        { }
};

struct fgmm_uv_t : fgmm_base_t< mfuv_t > {
    using base_type = fgmm_base_t< mfuv_t >;

public:
    static constexpr auto default_k = 1.5;

public:
    explicit fgmm_uv_t (
        size_t n = base_type::default_modes,
        double a = base_type::default_alpha,
        double v = base_type::default_variance,
        double t = base_type::default_variance_threshold,
        double w = base_type::default_weight_threshold,
        double k = default_k)
        : base_type (n, a, v, t, w, k, mfuv_t ())
        { }
};

} // namespace bs

#include <bs/fgmm.cc>

#endif // BS_FGMM_HPP
