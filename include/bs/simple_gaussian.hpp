#ifndef BS_SIMPLE_GAUSSIAN_HPP
#define BS_SIMPLE_GAUSSIAN_HPP

#include <bs/defs.hpp>
#include <bs/detail/base.hpp>

#include <vector>

#include <opencv2/core/mat.hpp>

namespace bs {

struct simple_gaussian_t : detail::base_t {
    explicit simple_gaussian_t (
        const cv::Mat&, double = .0001, double = .25, double = .67);

public:
    const cv::Mat&
    operator() (const cv::Mat&);

private:
    cv::Mat m_, v_;
    float alpha_, threshold_, noise_;
};

}

#endif // BS_SIMPLE_GAUSSIAN_HPP
