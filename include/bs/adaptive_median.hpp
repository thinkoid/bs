#ifndef BS_ADAPTIVE_MEDIAN_HPP
#define BS_ADAPTIVE_MEDIAN_HPP

#include <bs/defs.hpp>
#include <opencv2/core/mat.hpp>

namespace bs {

//
// @Article{McFarlane1995,
// author="McFarlane, N. J. B.
// and Schofield, C. P.",
// title="Segmentation and tracking of piglets in images",
// journal="Machine Vision and Applications",
// year="1995",
// volume="8",
// number="3",
// pages="187--193",
// issn="1432-1769",
// doi="10.1007/BF01215814",
// url="http://dx.doi.org/10.1007/BF01215814"
// }
//

struct adaptive_median {
    explicit adaptive_median (const cv::Mat&, size_t, size_t);

public:
    const cv::Mat&
    operator() (const cv::Mat&);

    const cv::Mat&
    mask () const {
        return mask_;
    }

private:
    cv::Mat background_, mask_;
    size_t frame_interval_, frame_counter_, threshold_;
};

}

#endif // BS_ADAPTIVE_MEDIAN_HPP
