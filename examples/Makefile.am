# -*- mode: makefile -*-

EXTRA_DIST = 

include $(top_srcdir)/Makefile.common

bin_PROGRAMS =                                  \
    adaptive_learning                           \
    adaptive_median                             \
    adaptive_selective_learning                 \
    moving_mean                                 \
    null                                        \
    sigma_delta                                 \
    static_frame                                \
    simple_frame                                \
    temporal_median                             \
    windowed_mean                               \
    windowed_variance


adaptive_learning_SOURCES = adaptive_learning.cpp options.cpp
adaptive_learning_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)

adaptive_median_SOURCES = adaptive_median.cpp options.cpp
adaptive_median_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)

adaptive_selective_learning_SOURCES = adaptive_selective_learning.cpp options.cpp
adaptive_selective_learning_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)

moving_mean_SOURCES = moving_mean.cpp options.cpp
moving_mean_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)

null_SOURCES = null.cpp options.cpp
null_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)

sigma_delta_SOURCES = sigma_delta.cpp options.cpp
sigma_delta_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)

static_frame_SOURCES = static_frame.cpp options.cpp
static_frame_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)

simple_frame_SOURCES = simple_frame.cpp options.cpp
simple_frame_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)

temporal_median_SOURCES = temporal_median.cpp options.cpp
temporal_median_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)

windowed_mean_SOURCES = windowed_mean.cpp options.cpp
windowed_mean_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)

windowed_variance_SOURCES = windowed_variance.cpp options.cpp
windowed_variance_LDADD = $(top_srcdir)/src/libbs.la $(LIBS)
