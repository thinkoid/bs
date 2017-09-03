#include <iostream>

#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <bs/frame_range.hpp>
#include <bs/utils.hpp>
#include <bs/sigma_delta.hpp>

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

    ("multiplier,m", po::value< double > ()->default_value (2.),
     "delta multiplier")

    ("min-variance,v", po::value< size_t > ()->default_value (2),
     "minimal variance threshold")

    ("max-variance,V", po::value< size_t > ()->default_value (255),
     "maximal variance threshold");

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
process_sigma_delta (cv::VideoCapture& cap, const options_t& opts) {
    const bool display = opts.have ("display");

    //
    // Bootstrap from the first frame:
    //
    cv::Mat background = bs::scale_frame (*bs::getframes_from (cap).begin ());

    bs::sigma_delta_t sigma_delta (
        background,
        opts ["multiplier"].as< double > (),
        opts ["min-variance"].as< size_t > (),
        opts ["max-variance"].as< size_t > ());

    for (auto& frame : bs::getframes_from (cap)) {
        bs::frame_delay temp { 10 };

        auto mask = sigma_delta (bs::scale_frame (frame));

        if (display)
            imshow ("Sigma-delta difference", mask);

        if (temp.wait_for_key (27))
            break;
    }
}

////////////////////////////////////////////////////////////////////////

int main (int argc, char** argv) {
    program_options_from (argc, argv);
    return run_with (process_sigma_delta, global_options ()), 0;
}
