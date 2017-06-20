#ifndef BS_WINDOWED_VARIANCE_HPP
#define BS_WINDOWED_VARIANCE_HPP

#include <bs/defs.hpp>

#include <opencv2/core/mat.hpp>
#include <boost/circular_buffer.hpp>

#include <vector>

namespace bs {

struct windowed_variance {
    explicit windowed_variance (std::vector< double >, size_t);

public:
    const cv::Mat&
    operator() (const cv::Mat&);

    const cv::Mat&
    mask () const {
        return mask_;
    }

private:
    cv::Mat mask_;
    boost::circular_buffer< cv::Mat > framebuf_;
    std::vector< double > weights_;
    size_t threshold_;
};

}

#endif // BS_WINDOWED_VARIANCE_HPP
