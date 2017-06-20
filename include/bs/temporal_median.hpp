#ifndef BS_TEMPORAL_MEDIAN_HPP
#define BS_TEMPORAL_MEDIAN_HPP

#include <bs/defs.hpp>

#include <opencv2/core/mat.hpp>
#include <boost/circular_buffer.hpp>

#include <vector>

namespace bs {

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

struct temporal_median_bootstrap {
    explicit temporal_median_bootstrap (
        size_t block_size = 16,
        size_t threshold = 15,
        size_t threshold_increment = 15,
        size_t motionthreshold = 5,
        size_t stablethreshold = 10,
        size_t thrash_limit = 2);

    bool operator() (const cv::Mat& frame) {
        return complete_ || process (frame);
    }

    cv::Mat background () const {
        return background_;
    }

private:
    bool
    initialize_background_from (const cv::Mat&);

    bool
    update_background_from (const cv::Mat&);

    bool
    process (const cv::Mat&);

    struct block {
        size_t x, y, w, h;
        size_t updates, motionthreshold;
        bool stable;
    };

private:
    cv::Mat background_;

    boost::circular_buffer< cv::Mat > framebuf_;

    std::vector< block > blocks_;

    size_t block_size_;
    size_t threshold_, threshold_increment_;
    size_t motionthreshold_, stablethreshold_;
    size_t thrashed_frames_, thrash_limit_;
    int init_, complete_;
};

struct temporal_median {
    explicit temporal_median (
        const cv::Mat&, size_t = 9, size_t = 16, double = 7,
        size_t = 2, size_t = 4);

    const cv::Mat&
    operator() (const cv::Mat&);

    const cv::Mat&
    mask () const {
        return mask_;
    }

private:
    cv::Mat
    threshold (const cv::Mat&, const cv::Mat&, size_t = 255);

    std::tuple< cv::Mat, cv::Mat, cv::Mat >
    calculate_masks () const;

    cv::Mat
    compose_masks (const cv::Mat&, const cv::Mat&, size_t = 255);

private:
    cv::Mat background_, mask_;
    boost::circular_buffer< cv::Mat > history_;

    double lambda_;
    size_t lo_, hi_, frame_interval_, frame_counter_;
};

} // namespace bs

#endif // BS_TEMPORAL_MEDIAN_HPP
