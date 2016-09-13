#ifndef BS_MOVING_VARIANCE_HPP
#define BS_MOVING_VARIANCE_HPP

#include <vector>

#include <boost/circular_buffer.hpp>

#include <opencv2/core/mat.hpp>

namespace bs {

struct moving_variance {
    explicit moving_variance (std::vector< double >, int);

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
    int threshold_;
};

}

#endif // BS_MOVING_VARIANCE_HPP
