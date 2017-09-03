# -*- mode: makefile -*-

EXTRA_DIST = 

include $(top_srcdir)/Makefile.common

bin_PROGRAMS =                                  \
    adaptive_median                             \
    fuzzy_choquet                               \
    fuzzy_sugeno                                \
    grimson_gmm                                 \
    sigma_delta                                 \
    simple_gaussian                             \
    temporal_median                             \
    zivkovic_gmm


adaptive_median_SOURCES = adaptive_median.cpp options.cpp
adaptive_median_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)

fuzzy_choquet_SOURCES = fuzzy_choquet.cpp options.cpp
fuzzy_choquet_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)

fuzzy_sugeno_SOURCES = fuzzy_sugeno.cpp options.cpp
fuzzy_sugeno_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)

grimson_gmm_SOURCES = grimson_gmm.cpp options.cpp
grimson_gmm_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)

sigma_delta_SOURCES = sigma_delta.cpp options.cpp
sigma_delta_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)

simple_gaussian_SOURCES = simple_gaussian.cpp options.cpp
simple_gaussian_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)

temporal_median_SOURCES = temporal_median.cpp options.cpp
temporal_median_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)

zivkovic_gmm_SOURCES = zivkovic_gmm.cpp options.cpp
zivkovic_gmm_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)
