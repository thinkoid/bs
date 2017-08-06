#include <bs/utils.hpp>
#include <bs/detail/lbp.hpp>

#include <opencv2/imgproc.hpp>

namespace bs {
namespace detail {

template< typename T >
inline cv::Mat
lbp (const cv::Mat& src) {
    auto dst = cv::Mat (src.size (), src.type (), cv::Scalar (0));

    for (int i = 1; i < src.rows - 1; ++i) {
        const T* p = src.ptr< T > (i - 1);
        const T* q = src.ptr< T > (i);
        const T* r = src.ptr< T > (i + 1);

        T* s = dst.ptr< T > (i);
        ++s;

        for (int j = 1; j < src.cols - 1; ++j, ++p, ++q, ++r, ++s) {
            T t = q [1];

            unsigned u =
                ((p [0] >= t) << 7) +
                ((p [1] >= t) << 6) +
                ((p [2] >= t) << 5) +
                ((q [0] >= t)) +
                ((q [2] >= t) << 4) +
                ((r [0] >= t) << 1) +
                ((r [1] >= t) << 2) +
                ((r [2] >= t) << 3);

            s [0] = u;
        }
    }

    return dst;
}

cv::Mat
lbp (const cv::Mat& src) {
    BOOST_ASSERT (1 == src.channels ());

    cv::Mat dst;

#define T(x, y) case x: dst = detail::lbp< y > (src); break

    switch (src.type ()) {
        T (CV_8UC1,  unsigned char);
        T (CV_32FC1,         float);
    default:
        throw std::invalid_argument ("unsupported type");
    }

#undef T

    return dst;
}

}}
