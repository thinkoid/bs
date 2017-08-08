#ifndef BS_FUZZY_CHOQUET_HPP
#define BS_FUZZY_CHOQUET_HPP

#include <bs/defs.hpp>
#include <bs/detail/base.hpp>

#include <vector>

#include <opencv2/core/mat.hpp>

namespace bs {

//
// @inproceedings{el2008fuzzy,
//   title={Fuzzy integral for moving object detection},
//   author={El Baf, Fida and Bouwmans, Thierry and Vachon, Bertrand},
//   booktitle={Fuzzy Systems, 2008. FUZZ-IEEE 2008.(IEEE World Congress on Computational Intelligence). IEEE International Conference on},
//   pages={1729--1736},
//   year={2008},
//   organization={IEEE}
// }
//

struct fuzzy_choquet_t : detail::base_t {
    explicit fuzzy_choquet_t (
        const cv::Mat&, double = .01, double = .67,
        const std::vector< double >& g = { .6, .3, .1 });

public:
    const cv::Mat&
    operator() (const cv::Mat&);

private:
    double alpha_, threshold_;
    std::vector< double > g_;
};

}

#endif // BS_FUZZY_CHOQUET_HPP
