#ifndef BS_TEMPORAL_MEDIAN_HPP
#define BS_TEMPORAL_MEDIAN_HPP

#include <vector>

#include <boost/circular_buffer.hpp>

#include <opencv2/core/mat.hpp>

namespace bs {

struct temporal_median_bootstrap {
    explicit temporal_median_bootstrap (
        size_t block_size = 16,
        size_t threshold = 15,
        size_t threshold_increment = 15,
        size_t motionhreshold = 5,
        size_t stablehreshold = 10,
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
        size_t updates, motionhreshold;
        bool stable;
    };

    cv::Mat background_;

    boost::circular_buffer< cv::Mat > framebuf_;

    std::vector< block > blocks_;

    size_t block_size_;
    size_t threshold_, threshold_increment_;
    size_t motionhreshold_, stablehreshold_;
    size_t thrashed_frames_, thrash_limit_;
    int init_, complete_;
};

struct temporal_median {
    explicit temporal_median (
        const cv::Mat&, double = 7, size_t = 9, size_t = 2, size_t = 4);

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
    combine_masks (const cv::Mat&, const cv::Mat&, size_t = 255);

private:
    cv::Mat background_, mask_;

    boost::circular_buffer< cv::Mat > history_;

    double lambda_;
    size_t lo_, hi_, history_size_, history_pos_, frame_counter_;
};

} // namespace bs

#endif // BS_TEMPORAL_MEDIAN_HPP
