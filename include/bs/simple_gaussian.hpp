#ifndef BS_SIMPLE_GAUSSIAN_HPP
#define BS_SIMPLE_GAUSSIAN_HPP

#include <bs/defs.hpp>
#include <bs/detail/base.hpp>

#include <vector>

#include <opencv2/core/mat.hpp>

namespace bs {

struct simple_gaussian_t : detail::base_t {
    explicit simple_gaussian_t (const cv::Mat&, float = .0001, float = .25);

public:
    const cv::Mat&
    operator() (const cv::Mat&);

private:
    cv::Mat m_, v_;
    float alpha_, threshold_;
};

}

#endif // BS_SIMPLE_GAUSSIAN_HPP
