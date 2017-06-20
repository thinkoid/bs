#ifndef BS_FRAME_HPP
#define BS_FRAME_HPP

#include <bs/defs.hpp>
#include <opencv2/core/mat.hpp>

namespace bs {

struct simple_frame {
    explicit simple_frame (const cv::Mat&, size_t);

public:
    const cv::Mat&
    operator() (const cv::Mat&);

    const cv::Mat&
    mask () const {
        return mask_;
    }

private:
    void init ();

private:
    cv::Mat background_, mask_;
    size_t threshold_;
};

}

#endif // BS_FRAME_HPP
