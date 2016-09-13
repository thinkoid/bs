#ifndef BS_LINE_RANGE_HPP
#define BS_LINE_RANGE_HPP

struct lines_iterator : boost::iterator_facade <
    lines_iterator, std::string, std::forward_iterator_tag > {

    lines_iterator () : psin_ { }, pstr_ { }, delim_ { } { }

    lines_iterator (
        std::istream* psin, std::string* pstr, char delim)
        : psin_(psin), pstr_(pstr), delim_(delim) {
        increment ();
    }

private:
    friend class boost::iterator_core_access;

    void increment () {
        if (!std::getline (*psin_, *pstr_, delim_))
            *this = lines_iterator { };
    }

    bool equal (lines_iterator const& that) const {
        return pstr_ == that.pstr_;
    }

    std::string& dereference() const {
        return *pstr_;
    }

    std::istream* psin_;
    std::string* pstr_;
    char delim_;
};

using lines_range_base = boost::iterator_range< lines_iterator >;

struct lines_range_data {
    std::string str_;
};

struct lines_range : private lines_range_data, lines_range_base {
    explicit lines_range (std::istream & sin, char delim = 'n')
        : lines_range_base (
            lines_iterator { &sin, &str_, delim },
            lines_iterator { })
    { }
};

static inline lines_range
getlines_from (std::istream& sin, char delim = '\n') {
    return lines_range { sin, delim };
}

#endif // BS_LINE_RANGE_HPP
