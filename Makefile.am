bin_PROGRAMS = boostSimd
ACLOCAL_AMFLAGS = -I m4
boostSimd_CPPFLAGS = $(BOOST_CPPFLAGS) -I/usr/local/opt/boost.simd/include
boostSimd_LDFLAGS = @LIBS@ $(BOOST_LDFLAGS)
boostSimd_LDADD = $(BOOST_TIMER_LIB) $(BOOST_SYSTEM_LIB)
boostSimd_SOURCES = main.cpp boostSimd.cpp
