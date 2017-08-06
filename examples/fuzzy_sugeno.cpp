#include <iostream>

#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <bs/frame_range.hpp>
#include <bs/utils.hpp>
#include <bs/fuzzy_sugeno.hpp>

#include <options.hpp>
#include <run.hpp>

namespace std {

template< typename T >
std::ostream&
operator<< (std::ostream& s, const std::vector< T >& v) {
    copy (v.begin (), v.end (), ostream_iterator< T > (s, ","));
    return s;
}

}

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

    ("input,i",   po::value< std::string > ()->default_value ("0"),
     "input (file or stream index).")

    ("measure,m", po::value< std::vector< double > > ()->multitoken ()
     ->default_value (std::vector< double > { .4, .3, .3 }),
     "lambda measure, sum must be 1")

    ("alpha,a", po::value< double > ()->default_value (.01),
     "alpha")

    ("background-alpha,b", po::value< double > ()->default_value (.1),
     "alpha for background learning")

    ("learning-frames,n", po::value< size_t > ()->default_value (10UL),
     "number of learning frames")

    ("threshold,t", po::value< double > ()->default_value (.67),
     "threshold value");

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
process_fuzzy_sugeno (cv::VideoCapture& cap, const options_t& opts) {
    const bool display = opts.have ("display");

    bs::fuzzy_sugeno_bootstrap_t fuzzy_sugeno_bootstrap;

    for (auto& frame : bs::getframes_from (cap))
        if (fuzzy_sugeno_bootstrap (frame))
            break;

    bs::fuzzy_sugeno_t fuzzy_sugeno (
        fuzzy_sugeno_bootstrap.background (),
        opts ["alpha"].as< double > (),
        opts ["threshold"].as< double > (),
        opts ["measure"].as< std::vector< double > > ());

    for (auto& frame : bs::getframes_from (cap)) {
        bs::frame_delay temp { 10 };

        const auto mask = fuzzy_sugeno (frame);

        if (display)
            imshow ("Fuzzy Sugeno filter", mask);

        if (temp.wait_for_key (27))
            break;
    }
}

////////////////////////////////////////////////////////////////////////

int main (int argc, char** argv) {
    program_options_from (argc, argv);
    return run_with (process_fuzzy_sugeno, global_options ()), 0;
}
