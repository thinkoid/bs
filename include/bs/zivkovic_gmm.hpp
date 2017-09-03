#ifndef BS_ZIVKOVIC_GMM_HPP
#define BS_ZIVKOVIC_GMM_HPP

#include <bs/defs.hpp>
#include <bs/detail/base.hpp>

#include <vector>

#include <opencv2/core/mat.hpp>

namespace bs {

//
// @inproceedings{zivkovic2004improved,
//   title={Improved adaptive Gaussian mixture model for background subtraction},
//   author={Zivkovic, Zoran},
//   booktitle={Pattern Recognition, 2004. ICPR 2004. Proceedings of the 17th International Conference on},
//   volume={2},
//   pages={28--31},
//   year={2004},
//   organization={IEEE}
// }
//

struct zivkovic_gmm_t : detail::base_t {
    static constexpr auto default_modes = 4.;
    static constexpr auto default_alpha = .005;
    static constexpr auto default_variance = 16.;
    static constexpr auto default_variance_threshold = 15.;
    static constexpr auto default_weight_threshold = .7;
    static constexpr auto default_bias = .05;

public:
    explicit zivkovic_gmm_t (
        size_t = default_modes,
        double = default_alpha,
        double = default_variance_threshold,
        double = default_variance,
        double = default_weight_threshold,
        double = default_bias);

public:
    const cv::Mat&
    operator() (const cv::Mat&);

private:
    struct gaussian_t {
        double v, w, s;
        cv::Vec3d m;
    };

    gaussian_t
    default_gaussian (const cv::Vec3b& = cv::Vec3b ());

private:
    size_t size_;
    double alpha_, variance_threshold_, variance_, weight_threshold_, bias_;
    std::vector< std::vector< gaussian_t > > g_;
};

}

#endif // BS_ZIVKOVIC_GMM_HPP
