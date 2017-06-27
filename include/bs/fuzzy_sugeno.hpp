#ifndef BS_FUZZY_SUGENO_HPP
#define BS_FUZZY_SUGENO_HPP

#include <bs/defs.hpp>
#include <opencv2/core/mat.hpp>

#include <vector>

namespace bs {

struct fuzzy_sugeno_bootstrap {
    explicit fuzzy_sugeno_bootstrap (double = .1, size_t = 10UL);

    bool operator() (const cv::Mat& frame) {
        return 0 == frame_counter_ || process (frame);
    }

    cv::Mat background () const {
        return background_;
    }

private:
    bool
    process (const cv::Mat&);

private:
    cv::Mat background_;
    double alpha_;
    size_t frame_counter_;
};

struct fuzzy_sugeno {
    explicit fuzzy_sugeno (
        const cv::Mat&,
        const std::vector< double >& g = { .4, .3, .3 },
        double = .01, double = .67);

public:
    const cv::Mat&
    operator() (const cv::Mat&);

    const cv::Mat&
    mask () const {
        return mask_;
    }

private:
    cv::Mat mask_, background_;
    std::vector< double > g_;
    double alpha_, threshold_;
};

}

#endif // BS_FUZZY_SUGENO_HPP
