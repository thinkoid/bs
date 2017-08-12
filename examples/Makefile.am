# -*- mode: makefile -*-

EXTRA_DIST = 

include $(top_srcdir)/Makefile.common

bin_PROGRAMS =                                  \
    adaptive_median                             \
    fuzzy_choquet                               \
    fuzzy_sugeno                                \
    sigma_delta                                 \
    simple_gaussian                             \
    temporal_median


adaptive_median_SOURCES = adaptive_median.cpp options.cpp
adaptive_median_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)

fuzzy_choquet_SOURCES = fuzzy_choquet.cpp options.cpp
fuzzy_choquet_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)

fuzzy_sugeno_SOURCES = fuzzy_sugeno.cpp options.cpp
fuzzy_sugeno_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)

sigma_delta_SOURCES = sigma_delta.cpp options.cpp
sigma_delta_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)

simple_gaussian_SOURCES = simple_gaussian.cpp options.cpp
simple_gaussian_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)

temporal_median_SOURCES = temporal_median.cpp options.cpp
temporal_median_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)
