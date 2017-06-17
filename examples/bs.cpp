#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <boost/assign.hpp>

#include <boost/range/adaptor/filtered.hpp>
#include <boost/range/adaptor/transformed.hpp>
using namespace boost::adaptors;

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <options.hpp>

#include <bs/utils.hpp>
#include <bs/adaptive_background.hpp>
#include <bs/adaptive_median.hpp>
#include <bs/adaptive_selective_background.hpp>
#include <bs/frame_range.hpp>
#include <bs/moving_mean.hpp>
#include <bs/moving_variance.hpp>
#include <bs/previous_frame.hpp>
#include <bs/static_frame.hpp>
#include <bs/sigma_delta.hpp>
#include <bs/temporal_median.hpp>
#include <bs/windowed_moving_mean.hpp>
#include <bs/null.hpp>

static const cv::Scalar hicolor (0, 255, 0);

namespace cv {

inline void
rectangle (
    Mat& m, const Point& p, const Size_< size_t >& s, const Scalar& color,
    int thickness = 1, int lineType = 8, int shift = 0) {
    const Point first  (p.x - s.width / 2, p.y - s.height / 2);
    const Point second (p.x + s.width / 2, p.y + s.height / 2);
    cv::rectangle (m, first, second, color, thickness, lineType, shift);
}

} // namespace cv

////////////////////////////////////////////////////////////////////////

static void
program_options_from (int& argc, char** argv) {
    bool complete_invocation = false;

    bs::options_t program_options (argc, argv);

    if (program_options.have ("version")) {
        std::cout << "OpenCV v3.1\n";
        complete_invocation = true;
    }

    if (program_options.have ("help")) {
        std::cout << program_options.description () << std::endl;
        complete_invocation = true;
    }

    if (complete_invocation)
        exit (0);

    bs::global_options (program_options);
}

////////////////////////////////////////////////////////////////////////

static void
process_null (cv::VideoCapture& cap, const bs::options_t& opts) {
    bs::null subtractor;

    const bool display = opts.have ("display");

    for (auto& frame : bs::getframes_from (cap)) {
        bs::frame_delay temp { 40 };

        auto mask = subtractor (frame);

        if (display)
            imshow ("Null difference", mask);

        if (temp.wait_for_key (27))
            break;
    }
}

////////////////////////////////////////////////////////////////////////

static void
process_frame_difference (
    cv::VideoCapture& cap, const bs::options_t& opts) {

    const bool display = opts.have ("display");

    cv::Mat background = bs::scale_frame (
        *bs::getframes_from (cap).begin ());

    bs::previous_frame subtractor (
        background,
        opts ["threshold"].as< int > ());

    for (auto& frame : bs::getframes_from (cap)) {
        bs::frame_delay temp { 40 };

        auto mask = subtractor (bs::scale_frame (frame));

        if (display)
            imshow ("Simple frame difference", mask);

        if (temp.wait_for_key (27))
            break;
    }
}

////////////////////////////////////////////////////////////////////////

static void
process_static_frame_difference (
    cv::VideoCapture& cap, const bs::options_t& opts) {

    cv::Mat background = bs::scale_frame (*bs::getframes_from (cap).begin ());

    bs::static_frame subtractor (
        background,
        opts ["threshold"].as< int > ());

    const bool display = opts.have ("display");

    for (auto& frame : bs::getframes_from (cap)) {
        bs::frame_delay temp { 40 };

        auto mask = subtractor (bs::scale_frame (frame));

        if (display)
            imshow ("Static frame difference", mask);

        if (temp.wait_for_key (27))
            break;
    }
}

////////////////////////////////////////////////////////////////////////

static void
process_sigma_delta (cv::VideoCapture& cap, const bs::options_t& opts) {
    cv::Mat background = bs::scale_frame (*bs::getframes_from (cap).begin ());

    bs::sigma_delta subtractor (
        background,
        opts ["lambda"].as< double > (),
        opts ["lo" ].as< size_t > (),
        opts ["hi" ].as< size_t > ());

    const bool display = opts.have ("display");

    for (auto& frame : bs::getframes_from (cap)) {
        bs::frame_delay temp { 40 };

        auto mask = subtractor (bs::scale_frame (frame));

        if (display)
            imshow ("Sigma-delta difference", mask);

        if (temp.wait_for_key (27))
            break;
    }
}

////////////////////////////////////////////////////////////////////////

static void
process_adaptive_background (
    cv::VideoCapture& cap, const bs::options_t& opts) {

    const bool display = opts.have ("display");

    //
    // Bootstrap from the first frame:
    //
    cv::Mat background = bs::scale_frame (*bs::getframes_from (cap).begin ());

    bs::adaptive_background subtractor (
        background,
        opts ["alpha"    ].as< double > (),
        opts ["threshold"].as< int > ());

    for (auto& frame : bs::getframes_from (cap)) {
        bs::frame_delay temp { 40 };

        auto mask = subtractor (bs::scale_frame (frame));

        if (display)
            imshow ("Adaptive background difference", mask);

        if (temp.wait_for_key (27))
            break;
    }
}

////////////////////////////////////////////////////////////////////////

static void
process_adaptive_selective_background (
    cv::VideoCapture& cap, const bs::options_t& opts) {

    const bool display = opts.have ("display");

    cv::Mat background = bs::scale_frame (*bs::getframes_from (cap).begin ());

    bs::adaptive_selective_background subtractor (
        background,
        opts ["alpha"    ].as< double > (),
        opts ["threshold"].as< int > ());

    for (auto& frame : bs::getframes_from (cap)) {
        bs::frame_delay temp { 40 };

        auto mask = subtractor (bs::scale_frame (frame));

        if (display)
            imshow ("Adaptive selective background difference", mask);

        if (temp.wait_for_key (27))
            break;
    }
}

////////////////////////////////////////////////////////////////////////

static void
process_moving_mean_background (
    cv::VideoCapture& cap, const bs::options_t& opts) {

    const bool display = opts.have ("display");

    cv::Mat initial_frame = bs::scale_frame (*bs::getframes_from (cap).begin ());

    bs::moving_mean subtractor (
        initial_frame,
        opts ["alpha"    ].as< double > (),
        opts ["threshold"].as< int > ());

    for (auto& frame : bs::getframes_from (cap)) {
        bs::frame_delay temp { 40 };

        auto mask = subtractor (bs::scale_frame (frame));

        if (display)
            imshow ("Moving mean difference", mask);

        if (temp.wait_for_key (27))
            break;
    }
}

////////////////////////////////////////////////////////////////////////

static void
process_windowed_moving_mean (cv::VideoCapture& cap, const bs::options_t& opts) {
    const bool display = opts.have ("display");

    bs::windowed_moving_mean subtractor (
        opts ["weights"].as< std::vector< double > > (),
        opts ["threshold"].as< int > ());

    for (auto& frame : bs::getframes_from (cap)) {
        bs::frame_delay temp { 40 };

        auto mask = subtractor (bs::scale_frame (frame));

        if (display)
            imshow ("Moving mean difference", mask);

        if (temp.wait_for_key (27))
            break;
    }
}

////////////////////////////////////////////////////////////////////////

static void
process_moving_variance (
    cv::VideoCapture& cap, const bs::options_t& opts) {
    const bool display = opts.have ("display");

    bs::moving_variance subtractor (
        opts ["weights"].as< std::vector< double > > (),
        opts ["threshold"].as< int > ());

    for (auto& frame : bs::getframes_from (cap)) {
        bs::frame_delay temp { 40 };

        auto mask = subtractor (frame);

        if (display)
            imshow ("Moving mean difference", mask);

        if (temp.wait_for_key (27))
            break;
    }
}

////////////////////////////////////////////////////////////////////////

static void
process_adaptive_median_background (
    cv::VideoCapture& cap, const bs::options_t& opts) {

    const bool display = opts.have ("display");

    //
    // Bootstrap from the first frame:
    //
    cv::Mat background = bs::scale_frame (*bs::getframes_from (cap).begin ());

    bs::adaptive_median subtractor (
        background,
        opts ["frame-interval"].as< size_t > (),
        opts ["threshold"].as< int > ());

    for (auto& frame : bs::getframes_from (cap)) {
        bs::frame_delay temp { 40 };

        auto mask = subtractor (bs::scale_frame (frame));

        if (display)
            imshow ("Adaptive median background difference", mask);

        if (temp.wait_for_key (27))
            break;
    }
}

////////////////////////////////////////////////////////////////////////

static void
process_temporal_median_background (
    cv::VideoCapture& cap, const bs::options_t& opts) {

    using namespace boost::adaptors;
    using namespace boost::assign;

    const bool display = opts.have ("display");

    bs::temporal_median_bootstrap bootstrap;

    for (auto& frame : bs::getframes_from (cap))
        if (bootstrap (bs::scale_frame (frame)))
            break;

    bs::temporal_median subtractor (
        bootstrap.background (),
        opts ["lambda" ].as< double > (),
        opts ["history-size" ].as< size_t > (),
        opts ["lo" ].as< size_t > (),
        opts ["hi" ].as< size_t > ());

    for (auto& frame : bs::getframes_from (cap)) {
        bs::frame_delay temp { 40 };

        auto mask = subtractor (bs::scale_frame (frame));

        if (display)
            imshow ("Temporal median background difference", mask);

        if (temp.wait_for_key (27))
            break;
    }
}

////////////////////////////////////////////////////////////////////////

static void
run_from_stream (cv::VideoCapture& cap, const bs::options_t& opts) {
    const std::string algorithm = opts.have ("algorithm")
        ? opts ["algorithm"].as< std::string > ()
        : "null";

    if (algorithm == "static-frame-difference") {
        process_static_frame_difference (cap, opts);
    }
    else if (algorithm == "sigma-delta") {
        process_sigma_delta (cap, opts);
    }
    else if (algorithm == "frame-difference") {
        process_frame_difference (cap, opts);
    }
    else if (algorithm == "windowed-moving-mean") {
        process_windowed_moving_mean (cap, opts);
    }
    else if (algorithm == "moving-variance") {
        process_moving_variance (cap, opts);
    }
    else if (algorithm == "adaptive-background") {
        process_adaptive_background (cap, opts);
    }
    else if (algorithm == "adaptive-selective-background") {
        process_adaptive_selective_background (cap, opts);
    }
    else if (algorithm == "moving-mean") {
        process_moving_mean_background (cap, opts);
    }
    else if (algorithm == "adaptive-median") {
        process_adaptive_median_background (cap, opts);
    }
    else if (algorithm == "temporal-median") {
        process_temporal_median_background (cap, opts);
    }
    else {
        process_null (cap, opts);
    }
}

static void
run_from_file_with (const bs::options_t& opts) {
    const fs::path& filename (opts ["input"].as< std::string > ());
    const fs::path ext = filename.extension ();

    if (ext == ".avi" || ext == ".mp4" || ext == ".mov") {
        cv::VideoCapture cap;

        if (cap.open (filename.generic_string ())) {
            cap.set (CV_CAP_PROP_FPS, 25);
            run_from_stream (cap, opts);
        }
    }
    else
        std::cerr << "unsupported file type" << std::endl;
}

static void
run_from_camera_with (const bs::options_t& opts) {
    cv::VideoCapture cap;

    const int stream = std::stoi (opts ["input"].as< std::string > ());

    if (cap.open (stream)) {
        cap.set (CV_CAP_PROP_FPS, 25);
        run_from_stream (cap, opts);
    }
}

static void
run_with (const bs::options_t& opts) {
    const auto input = opts ["input"].as< std::string > ();

    if (fs::exists (input))
        run_from_file_with (opts);
    else
        run_from_camera_with (opts);
}

////////////////////////////////////////////////////////////////////////

int main (int argc, char** argv) {
    program_options_from (argc, argv);
    return run_with (bs::global_options ()), 0;
}
