# -*- mode: makefile -*-

EXTRA_DIST = 

include $(top_srcdir)/Makefile.common

LIBS += $(BOOST_UNIT_TEST_FRAMEWORK_LIBS) $(top_srcdir)/src/libbs.la

if DARWIN
  # Because stupid
  LIBS += -lc++abi
endif

TESTS = threshold
check_PROGRAMS = threshold

threshold_SOURCES = threshold.cpp
threshold_LDADD = $(LIBS)

bin_PROGRAMS = lbp_perf

lbp_perf_SOURCES = lbp_perf.cpp
lbp_perf_LDADD = -lbenchmark

