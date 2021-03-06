#include <iostream>

#include <bs/ewma.hpp>
#include <bs/frame_range.hpp>
#include <bs/utils.hpp>
#include <bs/temporal_median.hpp>

#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <range/v3/front.hpp>
#include <range/v3/action/take.hpp>
#include <range/v3/view/take.hpp>

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

    ("input,i",   po::value< std::string > ()->default_value ("0"),
     "input (file or stream index).")

    ("lambda", po::value< double > ()->default_value (7),
     "a threshold multiplier (2)")

    ("history-size", po::value< size_t > ()->default_value (9),
     "length of frame history buffer (9)")

    ("frame-interval", po::value< size_t > ()->default_value (16),
     "interval between frames sampled for history buffer (16)")

    ("lo", po::value< size_t > ()->default_value (2),
     "distance from median for low threshold(2)")

    ("hi", po::value< size_t > ()->default_value (4),
     "distance from median for high threshold (4)");

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

static cv::Mat
bootstrap (cv::VideoCapture& cap) {
    using namespace ranges;

    //
    // The bootstrapping function:
    //
    bs::ewma_t f;

    //
    // Learn the background over 15 frames:
    //
    auto frames = bs::getframes_from (cap);

    for (auto frame : (frames | view::take (15))) {
        f (bs::scale_frame (frame));
    }

    return f.value ();
}

static void
process_temporal_median_background (cv::VideoCapture& cap, const options_t& opts) {
    const bool display = opts.have ("display");

    auto background_model = bootstrap (cap);

    bs::temporal_median_t temporal_median (
        background_model,
        opts ["history-size" ].as< size_t > (),
        opts ["frame-interval" ].as< size_t > (),
        opts ["lo" ].as< size_t > (),
        opts ["hi" ].as< size_t > ());

    for (auto& frame : bs::getframes_from (cap)) {
        bs::frame_delay temp { 10 };

        auto mask = temporal_median (bs::scale_frame (frame));

        if (display)
            imshow ("Temporal median", mask);

        if (temp.wait_for_key (27))
            break;
    }
}

////////////////////////////////////////////////////////////////////////

int main (int argc, char** argv) {
    program_options_from (argc, argv);
    return run_with (process_temporal_median_background, global_options ()), 0;
}
