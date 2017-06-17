#ifndef BS_SIGMA_DELTA_HPP
#define BS_SIGMA_DELTA_HPP

#include <bs/defs.hpp>
#include <opencv2/core/mat.hpp>

namespace bs {

//
// @article{Manzanera:2007:NMD:1222671.1222999,
//  author = {Manzanera, Antoine and Richefeu, Julien C.},
//  title = {A New Motion Detection Algorithm Based on {\$\Sigma\$}-{\$\Delta\$}
//  Background Estimation},
//  journal = {Pattern Recogn. Lett.},
//  issue_date = {February, 2007},
//  volume = {28},
//  number = {3},
//  month = feb,
//  year = {2007},
//  issn = {0167-8655},
//  pages = {320--328},
//  numpages = {9},
//  url = {http://dx.doi.org/10.1016/j.patrec.2006.04.007},
//  doi = {10.1016/j.patrec.2006.04.007},
//  acmid = {1222999},
//  publisher = {Elsevier Science Inc.},
//  address = {New York, NY, USA},
//  keywords = {Background estimation, Motion detection, Recursive filtering},
// }
//
// @inproceedings{lacassagne2009motion,
//   title={Motion detection: Fast and robust algorithms for embedded systems},
//   author={Lacassagne, Lionel and Manzanera, Antoine and Dupret, Antoine},
//   booktitle={Image Processing (ICIP), 2009 16th IEEE International Conference on},
//   pages={3265--3268},
//   year={2009},
//   organization={IEEE}
// }

struct sigma_delta {
    explicit sigma_delta (
        const cv::Mat&, size_t = 2, size_t = 2, size_t = 255);

public:
    const cv::Mat&
    operator() (const cv::Mat&);

    const cv::Mat&
    mask () const {
        return mask_;
    }

private:
    cv::Mat mask_, m_, d_, v_, q_;
    size_t n_, Vmin_, Vmax_;
};

}

#endif // BS_SIGMA_DELTA_HPP
