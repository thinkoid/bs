#ifndef BS_COMMON_HPP
#define BS_COMMON_HPP

#include <opencv2/imgproc.hpp>

namespace bs {

const int BACKGROUND = 0;
const int FOREGROUND = 255;

inline cv::Mat
float_from (
    const cv::Mat& frame, double scale = 1. / 255, double offset = 0) {
    cv::Mat dst;
    return frame.convertTo (dst, CV_32F, scale, offset), dst;
}

inline cv::Mat
mono_from (const cv::Mat& frame, double scale = 255, double offset = 0) {
    cv::Mat dst;
    return frame.convertTo (dst, CV_8U, scale, offset), dst;
}

inline cv::Mat
median_blur (const cv::Mat& src, int size_t = 3) {
    cv::Mat dst;
    return cv::medianBlur (src, dst, size_t), dst;
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
threshold (const cv::Mat& src, double threshold = 1.0,
           double maxval = 255.0, int type = CV_THRESH_BINARY) {
    cv::Mat dst;
    return cv::threshold (src, dst, threshold, maxval, type), dst;
}

inline cv::Mat
power_of (const cv::Mat& arg, const double power) {
    cv::Mat dst;
    return cv::pow (arg, power, dst), dst;
}

inline cv::Mat
absdiff (const cv::Mat& lhs, const cv::Mat& rhs) {
    cv::Mat dst;
    return cv::absdiff (lhs, rhs, dst), dst;
}

inline cv::Mat
square_of (const cv::Mat& arg) {
    return power_of (arg, 2.0);
}

inline cv::Mat
bitwise_and (const cv::Mat& lhs, const cv::Mat& rhs) {
    cv::Mat dst;
    return cv::bitwise_and (lhs, rhs, dst), dst;
}

inline cv::Mat
bitwise_and (const cv::Mat& lhs, const cv::Mat& rhs, const cv::Mat& mask) {
    cv::Mat dst;
    return cv::bitwise_and (lhs, rhs, dst, mask), dst;
}

inline cv::Mat
bitwise_not (cv::Mat src) {
    cv::Mat dst;
    return cv::bitwise_not (src, dst), dst;
}

inline cv::Mat
bitwise_not (cv::Mat src, const cv::Mat& mask) {
    return cv::bitwise_not (src, src, mask), src;
}

inline cv::Mat
convert (cv::Mat src, int t, double a = 1, double b = 0) {
    return src.convertTo (src, t, a, b), src;
}

inline cv::Mat
mask (const cv::Mat& src, const cv::Mat& mask) {
    cv::Mat dst;
    return src.copyTo (dst, mask), dst;
}

inline cv::Mat
flip (const cv::Mat& src, int how = 0) {
    cv::Mat dst;
    return cv::flip (src, dst, how), dst;
}

inline int
chebyshev (const cv::Vec3b& lhs, const cv::Vec3b& rhs) {
    int tmp [] = {
        std::abs (lhs [0] - rhs [0]),
        std::abs (lhs [1] - rhs [1]),
        std::abs (lhs [2] - rhs [2])
    };

    return (std::max)({ tmp [0], tmp [1], tmp [2] });
}

} // namespace bs

#endif // BS_COMMON_HPP
