#ifndef BS_FGMM_UM_HPP
#define BS_FGMM_UM_HPP

#include <bs/defs.hpp>
#include <bs/detail/base.hpp>

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

struct fgmm_um_t : detail::base_t {
    static constexpr auto default_modes = 4.;
    static constexpr auto default_alpha = .005;
    static constexpr auto default_variance = 16.;
    static constexpr auto default_variance_threshold = 2.5;
    static constexpr auto default_weight_threshold = .7;
    static constexpr auto default_k = 2.;

public:
    explicit fgmm_um_t (
        size_t = default_modes,
        double = default_alpha,
        double = default_variance,
        double = default_variance_threshold,
        double = default_weight_threshold,
        double = default_k);

public:
    const cv::Mat&
    operator() (const cv::Mat&);

private:
    struct gaussian_t {
        double v, s, w, g;
        cv::Vec3d m;
    };

    gaussian_t
    make_gaussian (const cv::Vec3b&, double);

    gaussian_t
    make_gaussian (const cv::Vec3b&, double, double);

    gaussian_t
    make_gaussian (const cv::Vec3b&, double, double, double);

private:
    size_t size_;
    double alpha_, variance_, variance_threshold_,weight_threshold_, k_;
    std::vector< std::vector< gaussian_t > > g_;
};

}

#endif // BS_FGMM_UM_HPP
