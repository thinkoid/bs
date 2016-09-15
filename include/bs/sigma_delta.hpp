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

struct sigma_delta {
    explicit sigma_delta (const cv::Mat&, int = 4, int = 25);

public:
    const cv::Mat&
    operator() (const cv::Mat&);

    const cv::Mat&
    mask () const {
        return mask_;
    }

private:
    cv::Mat mask_, m_, d_, v_;
    int n_, threshold_;
};

}

#endif // BS_SIGMA_DELTA_HPP
