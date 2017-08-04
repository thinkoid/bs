#ifndef BS_DETAIL_THRESHOLD_HPP
#define BS_DETAIL_THRESHOLD_HPP

#include <bs/defs.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace bs {
namespace detail {

template< typename T >
struct threshold_base {
    using value_type = T;

    explicit threshold_base (value_type threshold, value_type value)
        : threshold (threshold), value (value)
    { }

    value_type threshold, value;
};

template< typename T >
struct threshold_binary : threshold_base< T > {
    using  base_type = threshold_base< T >;
    using value_type = typename base_type::value_type;

    explicit threshold_binary (value_type threshold, value_type value)
        : base_type (threshold, value)
    { }

    value_type
    operator() (const T& arg) const {
        if (arg > base_type::threshold)
            return base_type::value;
        else
            return value_type { };
    }
};

template< typename T >
struct threshold_binary_inv : threshold_base< T > {
    using  base_type = threshold_base< T >;
    using value_type = typename base_type::value_type;

    explicit threshold_binary_inv (value_type threshold, value_type value)
        : base_type (threshold, value)
    { }

    value_type
    operator() (const T& arg) const {
        if (arg < base_type::threshold)
            return base_type::value;
        else
            return value_type { };
    }
};

template< typename T >
struct threshold_trunc : threshold_base< T > {
    using  base_type = threshold_base< T >;
    using value_type = typename base_type::value_type;

    explicit threshold_trunc (value_type threshold, value_type value)
        : base_type (threshold, value)
    { }

    value_type
    operator() (const T& arg) const {
        if (arg > base_type::threshold)
            return base_type::threshold;
        else
            return arg;
    }
};

template< typename T >
struct threshold_tozero : threshold_base< T > {
    using  base_type = threshold_base< T >;
    using value_type = typename base_type::value_type;

    explicit threshold_tozero (value_type threshold, value_type value)
        : base_type (threshold, value)
    { }

    value_type
    operator() (const T& arg) const {
        if (arg > base_type::threshold)
            return arg;
        else
            return value_type { };
    }
};

template< typename T >
struct threshold_tozero_inv : threshold_base< T > {
    using  base_type = threshold_base< T >;
    using value_type = typename base_type::value_type;

    explicit threshold_tozero_inv (value_type threshold, value_type value)
        : base_type (threshold, value)
    { }

    value_type
    operator() (const T& arg) const {
        if (arg < base_type::threshold)
            return arg;
        else
            return value_type { };
    }
};

template< typename T >
inline cv::Mat
threshold (const cv::Mat& src, const T& t) {
    using value_type = typename T::value_type;

    cv::Mat dst = src.clone ();

    size_t rows = dst.rows;
    size_t cols = dst.cols;

    if (dst.isContinuous ()) {
        cols *= rows;
        rows = 1;
    }

    for (size_t i = 0; i < rows; ++i) {
        value_type* p = reinterpret_cast< value_type* > (
                            dst.ptr< unsigned char > (i));

        for (size_t j = 0; j < cols; ++j)
            p [j] = t (p [j]);
    }

    return dst;
}

template< typename T >
inline cv::Mat
threshold (const cv::Mat& src, T threshold_, T value, int type) {
    switch (type) {
    case CV_THRESH_BINARY:
        return threshold (src, threshold_binary< T > (threshold_, value));

    case CV_THRESH_BINARY_INV:
        return threshold (src, threshold_binary_inv< T > (threshold_, value));

    case CV_THRESH_TRUNC:
        return threshold (src, threshold_trunc< T > (threshold_, value));

    case CV_THRESH_TOZERO:
        return threshold (src, threshold_tozero< T > (threshold_, value));

    case CV_THRESH_TOZERO_INV:
        return threshold (src, threshold_tozero_inv< T > (threshold_, value));

    default:
        throw std::invalid_argument ("unsupported thresholding type");
    }
}

}}

#endif // BS_DETAIL_THRESHOLD_HPP
