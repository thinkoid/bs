// -*- mode: c++ -*-

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE unique_resource

#include <bs/utils.hpp>

#include <boost/format.hpp>
using fmt = boost::format;

#include <boost/type_index.hpp>
#include <boost/test/unit_test.hpp>
namespace utf = boost::unit_test;

#include <iostream>
#include <exception>

BOOST_AUTO_TEST_SUITE(details)

BOOST_AUTO_TEST_CASE (threshold_test) {
    cv::Mat x = (cv::Mat_< int > (2, 2) << -1, 0, 1, 2);

    {
        cv::Mat y = bs::threshold (x, 0, 3, CV_THRESH_BINARY);
        BOOST_TEST (0 == y.at< int > (0, 0));
        BOOST_TEST (0 == y.at< int > (0, 1));
        BOOST_TEST (3 == y.at< int > (1, 0));
        BOOST_TEST (3 == y.at< int > (1, 1));
    }

    {
        cv::Mat y = bs::threshold (x, 0, 3, CV_THRESH_BINARY_INV);
        BOOST_TEST (3 == y.at< int > (0, 0));
        BOOST_TEST (0 == y.at< int > (0, 1));
        BOOST_TEST (0 == y.at< int > (1, 0));
        BOOST_TEST (0 == y.at< int > (1, 1));
    }

    {
        cv::Mat y = bs::threshold (x, 0, 3, CV_THRESH_TRUNC);
        BOOST_TEST (-1 == y.at< int > (0, 0));
        BOOST_TEST ( 0 == y.at< int > (0, 1));
        BOOST_TEST ( 0 == y.at< int > (1, 0));
        BOOST_TEST ( 0 == y.at< int > (1, 1));
    }

    {
        cv::Mat y = bs::threshold (x, 0, 3, CV_THRESH_TOZERO);
        BOOST_TEST (0 == y.at< int > (0, 0));
        BOOST_TEST (0 == y.at< int > (0, 1));
        BOOST_TEST (1 == y.at< int > (1, 0));
        BOOST_TEST (2 == y.at< int > (1, 1));
    }

    {
        cv::Mat y = bs::threshold (x, 0, 3, CV_THRESH_TOZERO_INV);
        BOOST_TEST (-1 == y.at< int > (0, 0));
        BOOST_TEST ( 0 == y.at< int > (0, 1));
        BOOST_TEST ( 0 == y.at< int > (1, 0));
        BOOST_TEST ( 0 == y.at< int > (1, 1));
    }
}

BOOST_AUTO_TEST_SUITE_END()
