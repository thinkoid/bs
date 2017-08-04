#ifndef BS_FUZZY_SUGENO_HPP
#define BS_FUZZY_SUGENO_HPP

#include <bs/defs.hpp>
#include <bs/detail/base.hpp>

#include <vector>

#include <opencv2/core/mat.hpp>

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

struct fuzzy_sugeno_bootstrap_t {
    explicit fuzzy_sugeno_bootstrap_t (double = .1, size_t = 10UL);

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

struct fuzzy_sugeno_t : detail::base_t {
    explicit fuzzy_sugeno_t (
        const cv::Mat&, double = .01, double = .67,
        const std::vector< double >& g = { .4, .3, .3 });

public:
    const cv::Mat&
    operator() (const cv::Mat&);

private:
    double alpha_, threshold_;
    std::vector< double > g_;
};

}

#endif // BS_FUZZY_SUGENO_HPP
