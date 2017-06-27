#ifndef BS_FUZZY_SUGENO_HPP
#define BS_FUZZY_SUGENO_HPP

#include <bs/defs.hpp>
#include <opencv2/core/mat.hpp>

#include <vector>

namespace bs {

//
// @inproceedings{zhang:2006:FCT:2092283.2092396,
//  author = {zhang, Hongxun and xu, De},
//  title = {Fusing Color and Texture Features for Background Model},
//  booktitle = {Proceedings of the Third International Conference on Fuzzy Systems and Knowledge Discovery},
//  series = {FSKD'06},
//  year = {2006},
//  isbn = {3-540-45916-2, 978-3-540-45916-3},
//  location = {Xi'an, China},
//  pages = {887--893},
//  numpages = {7},
//  url = {http://dx.doi.org/10.1007/11881599_110},
//  doi = {10.1007/11881599_110},
//  acmid = {2092396},
//  publisher = {Springer-Verlag},
//  address = {Berlin, Heidelberg},
// }
//

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
