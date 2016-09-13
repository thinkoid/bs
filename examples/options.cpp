// -*- mode: c++ -*-

#include <options.hpp>

#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <fstream>
#include <functional>

namespace po = boost::program_options;

namespace bs {

static options_t global_options_;

const options_t&
global_options () {
    return global_options_;
}

options_t
global_options (options_t arg) {
    return std::swap (arg, global_options_), std::move (arg);
}

////////////////////////////////////////////////////////////////////////

options_t::options_t () { }

options_t::options_t (int argc, char** argv) {
    {
        auto tmp = std::make_pair (
            "program", po::variable_value (std::string (argv [0]), false));
        map_.insert (tmp);
    }

    std::vector< std::string > config_files;

    po::options_description generic ("Generic options");
    po::options_description config ("Configuration options");

    generic.add_options ()
        ("version,v", "version")
        ("help,h", "this");

    config.add_options ()
        ("display,d", "display frames.")

        ("input,i",   po::value< std::string > ()->default_value ("0"),
         "input (file or stream index).")

        ("algorithm,a", po::value< std::string > (),
         "algorithm (adaptive-background, adaptive-median, "
         "adaptive-selective-background, frame-difference, "
         "moving-mean, moving-variance, static-frame-difference, "
         "temporal-median, windowed-moving-mean)")

        ("alpha", po::value< double > ()->default_value (0.05),
         "alpha parameter (meaning depends on the algorithm, "
         "usually a weight)")

        ("lambda", po::value< double > ()->default_value (7.),
         "lambda parameter (meaning depends on the algorithm, "
         "usually a multiplier)")

        ("history-size", po::value< size_t > ()->default_value (9),
         "length of historical frame buffer")

        ("lo", po::value< size_t > ()->default_value (2),
         "distance from median for low threshold")

        ("hi", po::value< size_t > ()->default_value (4),
         "distance from median for high threshold")

        ("fps", po::value< int > ()->default_value (25),
         "fps (meaning depends on the algorithm)")

        ("frame-interval", po::value< size_t > ()->default_value (25),
         "frame-interval (interval between samples)")

        ("weights", po::value< std::vector< double > > ()->multitoken ()
         ->default_value (std::vector< double >{ .33, .33, .33 }, ""),
         "weigths (meaning depends on the algorithm, usually a multiplier)")

        ("threshold,t", po::value< int > ()->default_value (15),
         "binarization threshold");

    desc_ = boost::make_shared< po::options_description > ();

    desc_->add (generic);
    desc_->add (config);

    store (
        po::command_line_parser (argc, argv).options (*desc_).run (),
        map_);

    notify (map_);
}

options_t::const_reference
options_t::operator[] (const char* s) const {
    return map_ [s];
}

bool
options_t::have (const char* key) const {
    return bool (map_.count (key));
}

} // namespace bs
