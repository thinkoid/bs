// -*- mode: c++ -*-

#include <bs/lbp.hpp>
#include <bs/frame_range.hpp>
#include <bs/utils.hpp>

#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <boost/timer/timer.hpp>

#include <iostream>
#include <exception>

////////////////////////////////////////////////////////////////////////

static void
run_from_stream (cv::VideoCapture& cap) {
    for (auto& frame : bs::getframes_from (cap)) {
        bs::frame_delay temp { 0 };

        cv::Mat scaled_frame = bs::scale_frame (frame), output_frame;

        {
            using namespace boost::timer;

            auto_cpu_timer timer;
            output_frame = bs::lbp (scaled_frame);
        }

        imshow ("Original LBP algorithm", output_frame);

        if (temp.wait_for_key (27))
            break;
    }
}

static void
run_from_file (const char* s) {
    const fs::path& filename (s);
    const fs::path ext = filename.extension ();

    if (ext == ".avi" || ext == ".mp4" || ext == ".mov") {
        cv::VideoCapture cap;

        if (cap.open (filename.generic_string ())) {
            cap.set (CV_CAP_PROP_FPS, 25);
            run_from_stream (cap);
        }
    }
    else
        std::cerr << "unsupported file type" << std::endl;
}

static void
run_from_camera () {
    cv::VideoCapture cap;

    if (cap.open (0)) {
        cap.set (CV_CAP_PROP_FPS, 25);
        run_from_stream (cap);
    }
}

static void
run_with (const char* s) {
    if (fs::exists (s))
        run_from_file (s);
    else
        run_from_camera ();
}

////////////////////////////////////////////////////////////////////////

int main (int, char** argv) {
    return run_with (argv [1]), 0;
}
