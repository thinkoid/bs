#include <bs/utils.hpp>
#include <bs/detail/lbp.hpp>

#include <opencv2/imgproc.hpp>

namespace {

template< typename T >
inline cv::Mat
do_lbp (const cv::Mat& src, T off = { }) {
    auto dst = cv::Mat (src.size (), src.type (), cv::Scalar (0));

#pragma omp parallel for
    for (int i = 1; i < src.rows - 1; ++i) {
        const T* p = src.ptr< T > (i - 1);
        const T* q = src.ptr< T > (i);
        const T* r = src.ptr< T > (i + 1);

        T* s = dst.ptr< T > (i);
        ++s;

        for (int j = 1; j < src.cols - 1; ++j, ++p, ++q, ++r, ++s) {
            T t = q [1] + off;

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

}

namespace bs {

cv::Mat
lbp (const cv::Mat& src) {
    BOOST_ASSERT (1 == src.channels ());

#define T(x, y, z) case x: return do_lbp< y > (src, z)

    switch (src.type ()) {
        T (CV_8UC1,  unsigned char, 0);
        T (CV_32FC1,         float, 1./255);
    default:
        throw std::invalid_argument ("unsupported type");
    }

#undef T
}

}
