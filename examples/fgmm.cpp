#include <iostream>

#include <bs/ewma.hpp>
#include <bs/frame_range.hpp>
#include <bs/fgmm.hpp>
#include <bs/utils.hpp>

#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <range/v3/front.hpp>
#include <range/v3/action/take.hpp>
#include <range/v3/view/take.hpp>

#include <opencv2/highgui.hpp>
using namespace cv;

#include <options.hpp>
#include <run.hpp>

//
// options_t::options_t is specific to each example:
//
options_t::options_t (int argc, char** argv) {
    {
        auto tmp = std::make_pair (
            "program", po::variable_value (std::string (argv [0]), false));
        map_.insert (tmp);
    }

    po::options_description generic ("Generic options");
    po::options_description config ("Configuration options");

    generic.add_options ()
    ("version", "version")
    ("help", "this");

    config.add_options ()
    ("display,d", "display frames.")

    ("input,i", po::value< std::string > ()->default_value ("0"),
     "input (file or stream index).")

    ("size,s", po::value< size_t > ()->default_value (4UL),
     "maximum number of distributions.")

    ("alpha,a", po::value< double > ()->default_value (.005),
     "learning alpha.")

    ("variance,v", po::value< double > ()->default_value (16.),
     "default variance for new distributions.")

    ("variance-threshold,t", po::value< double > ()->default_value (2.5),
     "variance threshold for matching a distribution.")

    ("weight-threshold,w", po::value< double > ()->default_value (.7),
     "maximum weight (probability) for likely background distributions.")

    ("km,k", po::value< double > ()->default_value (2.),
     "uncertainty factor.");

    desc_ = boost::make_shared< po::options_description > ();

    desc_->add (generic);
    desc_->add (config);

    store (po::command_line_parser (argc, argv).options (*desc_).run (), map_);

    notify (map_);
}

////////////////////////////////////////////////////////////////////////

static void
program_options_from (int& argc, char** argv) {
    bool complete_invocation = false;

    options_t program_options (argc, argv);

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

    global_options (program_options);
}

////////////////////////////////////////////////////////////////////////

static void
process_fgmm (cv::VideoCapture& cap, const options_t& opts) {
    const bool display = opts.have ("display");

    bs::fgmm_uv_t op (
        opts ["size"].as< size_t > (),
        opts ["alpha"].as< double > (),
        opts ["variance"].as< double > (),
        opts ["variance-threshold"].as< double > (),
        opts ["weight-threshold"].as< double > (),
        opts ["km"].as< double > ());

    namedWindow ("Type 2 Fuzzy GMM");
    namedWindow ("Type 2 Fuzzy GMM background");

    moveWindow ("Type 2 Fuzzy GMM", 0, 0);
    moveWindow ("Type 2 Fuzzy GMM background", 512, 0);

    for (auto& frame : bs::getframes_from (cap)) {
        bs::frame_delay temp { 0 };

        auto src = bs::resize_frame (frame, 512. / frame.cols);

        if (display) {
            imshow ("Type 2 Fuzzy GMM", op (src));
            imshow ("Type 2 Fuzzy GMM background", op.background ());
        }

        if (temp.wait_for_key (27))
            break;
    }
}

////////////////////////////////////////////////////////////////////////

int main (int argc, char** argv) {
    program_options_from (argc, argv);
    return run_with (process_fgmm, global_options ()), 0;
}
