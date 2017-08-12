#include <iostream>

#include <bs/ewma.hpp>
#include <bs/frame_range.hpp>
#include <bs/simple_gaussian.hpp>
#include <bs/utils.hpp>

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
    ("version,v", "version")
    ("help,h", "this");

    config.add_options ()
    ("display,d", "display frames.")

    ("input,i", po::value< std::string > ()->default_value ("0"),
     "input (file or stream index).")

    ("alpha,a", po::value< double > ()->default_value (.0001),
     "Learning alpha.")

    ("threshold,t", po::value< double > ()->default_value (.25),
     "Mahalanobis distance threshold.")

    ("noise,n", po::value< double > ()->default_value (.67),
     "Maximum noise.");

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
    // Learn the background over 30 frames:
    //
    auto frames = bs::getframes_from (cap);

    for (auto frame : (frames | view::take (30))) {
        f (frame);
    }

    return f.value ();
}

static void
process_simple_gaussian (cv::VideoCapture& cap, const options_t& opts) {
    const bool display = opts.have ("display");

    auto background_model = bootstrap (cap);

    bs::simple_gaussian_t simple_gaussian (
        background_model,
        opts ["alpha"].as< double > (),
        opts ["threshold"].as< double > (),
        opts ["noise"].as< double > ());

    for (auto& frame : bs::getframes_from (cap)) {
        bs::frame_delay temp { 0 };

        auto mask = simple_gaussian (frame);

        if (display)
            imshow ("Simple Gaussian filter", mask);

        if (temp.wait_for_key (27))
            break;
    }
}

////////////////////////////////////////////////////////////////////////

int main (int argc, char** argv) {
    program_options_from (argc, argv);
    return run_with (process_simple_gaussian, global_options ()), 0;
}
