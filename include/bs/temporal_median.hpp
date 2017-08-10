#ifndef BS_TEMPORAL_MEDIAN_HPP
#define BS_TEMPORAL_MEDIAN_HPP

#include <bs/defs.hpp>
#include <bs/detail/base.hpp>

#include <opencv2/core/mat.hpp>
#include <boost/circular_buffer.hpp>

#include <vector>

namespace bs {

//
// Mostly based on the algorithm described in:
//
// @inproceedings{Calderara:2006:RBS:1178782.1178814,
//  author = {Calderara, Simone and Melli, Rudy and Prati, Andrea and Cucchiara,
//  Rita},
//  title = {Reliable Background Suppression for Complex Scenes},
//  booktitle = {Proceedings of the 4th ACM International Workshop on Video
//  Surveillance and Sensor Networks},
//  series = {VSSN '06},
//  year = {2006},
//  isbn = {1-59593-496-0},
//  location = {Santa Barbara, California, USA},
//  pages = {211--214},
//  numpages = {4},
//  url = {http://doi.acm.org/10.1145/1178782.1178814},
//  doi = {10.1145/1178782.1178814},
//  acmid = {1178814},
//  publisher = {ACM},
//  address = {New York, NY, USA},
//  keywords = {background suppression, people detection and tracking, shadow
//  detection},
// }
//

struct temporal_median_t : detail::base_t {
    explicit temporal_median_t (
        const cv::Mat&, size_t = 9, size_t = 16, size_t = 30, size_t = 60);

    const cv::Mat&
    operator() (const cv::Mat&);

private:
    cv::Mat
    calculate_median () const;

    cv::Mat
    merge_masks (const cv::Mat&, const cv::Mat&);

private:
    boost::circular_buffer< cv::Mat > history_;
    size_t lo_, hi_, frame_interval_, frame_counter_;
};

}

#endif // BS_TEMPORAL_MEDIAN_HPP
