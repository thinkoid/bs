# -*- mode: autoconf -*-

AC_INIT([bs],[0.0.1],[])

AC_CONFIG_MACRO_DIR([etc/m4])
AC_CONFIG_AUX_DIR([etc/m4])

AM_INIT_AUTOMAKE([-Wno-portability subdir-objects dist-bzip2 nostdinc foreign])
AC_CONFIG_HEADERS([include/bs/_config.hpp:include/bs/config.ac])
AM_SILENT_RULES([yes])

AC_PROG_CXX
AC_CONFIG_CXX_WARNINGS

AC_ENABLE_CXX_DIALECT([c++1z])
AC_ENABLE_OPENMP

LT_INIT
AC_PROG_MAKE_SET(gmake)
AM_PROG_AS

AC_CONFIG_WITH_RANGE3

BOOST_REQUIRE([1.69.0])

BOOST_FILESYSTEM
BOOST_PROGRAM_OPTIONS
BOOST_SYSTEM
BOOST_TEST

AC_PKG_CONFIG_WITH([opencv4])

AM_CONDITIONAL([DARWIN],[test `uname` == Darwin])
AM_CONDITIONAL([LINUX], [test `uname` == Linux])

AC_CONFIG_FILES(Makefile)
AC_CONFIG_FILES(etc/Makefile)
AC_CONFIG_FILES(include/Makefile)
AC_CONFIG_FILES(src/Makefile)
AC_CONFIG_FILES(examples/Makefile)
AC_CONFIG_FILES(tests/Makefile)

AC_OUTPUT

