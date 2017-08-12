#include <bs/utils.hpp>
#include <bs/simple_gaussian.hpp>

#include <opencv2/imgproc.hpp>

#include <iostream>
using namespace std;

namespace bs {

/* explicit */
simple_gaussian_t::simple_gaussian_t (
    const cv::Mat& b, double a, double t, double n)
    : detail::base_t (b, { b.size (), CV_8U, cv::Scalar (0) }),
      m_ (bs::float_from (b)), v_ (b.size (), CV_32FC3, { n, n, n }),
      alpha_ (a), threshold_ (t), noise_ (n)
{ }

const cv::Mat&
simple_gaussian_t::operator() (const cv::Mat& frame) {
    auto mul = [](const cv::Vec3f& a, const cv::Vec3f& b) -> cv::Vec3f {
        return { a [0] * b [0] + a [1] * b [1] + a [2] * b [2] };
    };

    for (size_t i = 0; i < frame.total (); ++i) {
        const auto& src = cv::Vec3f (frame.at< cv::Vec3b > (i)) / 255;

        auto& m = m_.at< cv::Vec3f > (i);
        auto& v = v_.at< cv::Vec3f > (i);

        double a, b, c;

        a = src [0] - m [0];
        b = src [1] - m [1];
        c = src [2] - m [2];

        //
        // Mahalanobis distance:
        //
        double squared_distance =
            a * a / v [0] +
            b * b / v [1] +
            c * c / v [2];

        mask_.at< unsigned char > (i) = squared_distance > threshold_ ? 255 : 0;

        //
        // Rolling mean and variance:
        //
        {
            auto diff = src - m;

            auto inc = alpha_ * diff;
            m += inc;

            v = (1 - alpha_) * (v + mul (diff, inc));

            v [0] = (std::min) (v [0], noise_);
            v [1] = (std::min) (v [1], noise_);
            v [2] = (std::min) (v [2], noise_);
        }

        //
        // Update the background:
        //
        auto& bg = background_.at< cv::Vec3b > (i);

        bg [0] = cv::saturate_cast< unsigned char > (m [0] * 255);
        bg [1] = cv::saturate_cast< unsigned char > (m [1] * 255);
        bg [2] = cv::saturate_cast< unsigned char > (m [2] * 255);
    }

    return mask_;
}

}
