#include <bs/utils.hpp>
#include <bs/simple_gaussian.hpp>

#include <opencv2/imgproc.hpp>

#include <iostream>
using namespace std;

namespace bs {

/* explicit */
simple_gaussian_t::simple_gaussian_t (
    const cv::Mat& b, float a, float t)
    : detail::base_t (b, { b.size (), CV_8U, cv::Scalar (0) }),
      m_ (bs::float_from (b)), v_ (b.size (), CV_32FC3, { .6, .6, .6 }),
      alpha_ (a), threshold_ (t * t)
{ }

const cv::Mat&
simple_gaussian_t::operator() (const cv::Mat& frame) {
    auto mul = [](const cv::Vec3f& a, const cv::Vec3f& b) -> cv::Vec3f {
        return { a [0] * b [0] + a [1] * b [1] + a [2] * b [2] };
    };

#pragma omp parallel for
    for (size_t i = 0; i < frame.total (); ++i) {
        const auto& src = cv::Vec3f (frame.at< cv::Vec3b > (i)) / 255;

        auto& m = m_.at< cv::Vec3f > (i);
        auto& v = v_.at< cv::Vec3f > (i);

        float a, b, c;

        a = src [0] - m [0];
        b = src [1] - m [1];
        c = src [2] - m [2];

        //
        // Squared normalized Euclidean distance:
        //
        const float distance =
            a * a / v [0] +
            b * b / v [1] +
            c * c / v [2];

        mask_.at< unsigned char > (i) = distance > threshold_ ? 255 : 0;

        //
        // Rolling mean and variance:
        //
        {
            auto diff = src - m;

            auto inc = alpha_ * diff;
            m += inc;

            v = (1 - alpha_) * (v + mul (diff, inc));
        }

        //
        // Update the background:
        //
        background_.at< cv::Vec3b > (i) = cv::Vec3b (m * 255);
    }

    return mask_;
}

}
