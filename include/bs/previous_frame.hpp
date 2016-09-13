#ifndef BS_FRAME_HPP
#define BS_FRAME_HPP

#include <opencv2/core/mat.hpp>

namespace bs {

struct previous_frame {
    explicit previous_frame (const cv::Mat&, int);

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
    int threshold_;
};

}

#endif // BS_FRAME_HPP
