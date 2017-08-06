#ifndef BS_UTILS_HPP
#define BS_UTILS_HPP

#include <bs/defs.hpp>
#include <bs/detail/threshold.hpp>

#include <chrono>
#include <functional>

#include <opencv2/imgproc.hpp>

namespace bs {
namespace detail {

inline cv::Mat
scale_frame (cv::Mat& frame, double factor) {
    cv::Mat bw;
    cv::cvtColor (frame, bw, cv::COLOR_BGR2GRAY);

    cv::Mat tiny;
    cv::resize (bw, tiny, cv::Size (), factor, factor, cv::INTER_LINEAR);

    return tiny;
}

}

inline cv::Mat
border (const cv::Mat& src, size_t n = 1) {
    cv::Mat dst;
    cv::copyMakeBorder (
        src, dst, n, n, n, n, cv::BORDER_CONSTANT, 0);
    return dst;
}

inline cv::Mat
unborder (const cv::Mat& src, size_t n = 1) {
    return cv::Mat (
        src (cv::Rect (n, n, src.cols - 2 * n, src.rows - 2 * n)));
}

inline std::pair< double, double >
minmax (const cv::Mat& src) {
    double a, b;
    minMaxLoc (src, &a, &b);
    return { a, b };
}

inline cv::Mat
convert (const cv::Mat& src, int t, double a = 1, double b = 0) {
    cv::Mat dst;
    return src.convertTo (dst, t, a, b), dst;
}

inline cv::Mat
float_from (const cv::Mat& src, double scale = 1. / 255, double offset = 0.) {
    return convert (src, CV_32F, scale, offset);
}

inline cv::Mat
mono_from (const cv::Mat& src, double scale = 255., double offset = 0.) {
    return convert (src, CV_8U, scale, offset);
}

inline cv::Mat
median_blur (const cv::Mat& src, int size = 3) {
    cv::Mat dst;
    return cv::medianBlur (src, dst, size), dst;
}

inline cv::Mat
multiply (const cv::Mat& lhs, const cv::Mat& rhs) {
    cv::Mat dst;
    return cv::multiply (lhs, rhs, dst), dst;
}

inline cv::Mat
convert_color (const cv::Mat& src, int type) {
    cv::Mat dst;
    return cv::cvtColor (src, dst, type), dst;
}

inline int
gray_from (int r, int g, int b) {
    return double (r + g + b) / 3;
}

inline int
gray_from (const cv::Vec3b& arg) {
    return gray_from (arg [0], arg [1], arg [2]);
}

inline cv::Mat
gray_from (const cv::Mat& src) {
    return convert_color (src, cv::COLOR_BGR2GRAY);
}

inline cv::Mat
power_of (const cv::Mat& src, double power) {
    cv::Mat dst;
    return cv::pow (src, power, dst), dst;
}

inline cv::Mat
absdiff (const cv::Mat& lhs, const cv::Mat& rhs) {
    cv::Mat dst;
    return cv::absdiff (lhs, rhs, dst), dst;
}

inline cv::Mat
scale_frame (cv::Mat& src, size_t to = 512) {
    return detail::scale_frame (src, double (to) / src.cols);
}

inline cv::Mat
threshold (const cv::Mat& src, double threshold_ = 1., double maxval = 255.,
           int type = cv::THRESH_BINARY) {
    cv::Mat dst;

    switch (src.type ()) {
    case CV_8U:
    case CV_32F:
        cv::threshold (src, dst, threshold_, maxval, type);
        break;

#define T(x, y) case x:                                                 \
        dst = detail::threshold< y > (src, threshold_, maxval, type);   \
        break

        T (CV_16U, unsigned short int);
        T (CV_16S, short int);
        T (CV_32S, int);

#undef T

    default:
        throw std::invalid_argument ("unsupported array type");
    }

    return dst;
}

struct frame_delay {
    frame_delay (size_t value = 40)
        : value_ (value),
          begin_ (std::chrono::high_resolution_clock::now ())
    { }

    bool wait_for_key (int key) const  {
        using namespace std::chrono;

        int passed = duration_cast< milliseconds > (
                         high_resolution_clock::now () - begin_).count ();

        int remaining = value_ - passed;

        if (remaining < 1)
            remaining = 1;

        return key == cv::waitKey (remaining);
    }

private:
    int value_;
    std::chrono::high_resolution_clock::time_point begin_;
};

}

#endif // BS_UTILS_HPP
