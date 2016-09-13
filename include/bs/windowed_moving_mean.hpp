#ifndef BS_WINDOWED_MOVING_MEAN_HPP
#define BS_WINDOWED_MOVING_MEAN_HPP

#include <bs/defs.hpp>

#include <opencv2/core/mat.hpp>
#include <boost/circular_buffer.hpp>

#include <vector>

namespace bs {

struct windowed_moving_mean {
    explicit windowed_moving_mean (std::vector< double >, int);

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

#endif // BS_WINDOWED_MOVING_MEAN_HPP
