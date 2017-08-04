#ifndef BS_DETAIL_BASE_HPP
#define BS_DETAIL_BASE_HPP

#include <bs/defs.hpp>
#include <opencv2/core/mat.hpp>

namespace bs {
namespace detail {

struct base_t {
    explicit base_t (const cv::Mat& background = { }, const cv::Mat& mask = { })
        : background_ (background), mask_ (mask)
    { }

public:
    const cv::Mat&
    mask () const {
        return mask_;
    }

    const cv::Mat&
    background () const {
        return background_;
    }

protected:
    cv::Mat background_, mask_;
};

}}

#endif // BS_DETAIL_BASE_HPP
