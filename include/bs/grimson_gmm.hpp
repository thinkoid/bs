#ifndef BS_GRIMSON_GMM_HPP
#define BS_GRIMSON_GMM_HPP

#include <bs/defs.hpp>
#include <bs/detail/base.hpp>

#include <vector>

#include <opencv2/core/mat.hpp>

namespace bs {

//
// @inproceedings{stauffer1999adaptive,
//   title={Adaptive background mixture models for real-time tracking},
//   author={Stauffer, Chris and Grimson, W Eric L},
//   booktitle={Computer Vision and Pattern Recognition, 1999. IEEE Computer Society Conference on.},
//   volume={2},
//   pages={246--252},
//   year={1999},
//   organization={IEEE}
// }
//

struct grimson_gmm_t : detail::base_t {
    static constexpr auto default_modes = 4.;
    static constexpr auto default_alpha = .005;
    static constexpr auto default_variance = 16.;
    static constexpr auto default_variance_threshold = 15.;
    static constexpr auto default_weight_threshold = .7;

public:
    explicit grimson_gmm_t (
        size_t = default_modes,
        double = default_alpha,
        double = default_variance_threshold,
        double = default_variance,
        double = default_weight_threshold);

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
    double alpha_, variance_threshold_, variance_, weight_threshold_;
    std::vector< std::vector< gaussian_t > > g_;
};

}

#endif // BS_GRIMSON_GMM_HPP
