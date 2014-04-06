AC_PREREQ([2.57])

AC_INIT([boostSimd],[0.1.0])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_HEADERS([config.h])
AM_INIT_AUTOMAKE([0.1.0 foreign dist-bzip2 dist-xz])

LT_INIT([dlopen])
LT_PREREQ([1.0])

AC_LANG(C++)
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_LIBTOOL

AC_HEADER_RESOLV

AX_BOOST_BASE([1.48.0])
AX_BOOST_TIMER
AX_BOOST_SYSTEM

CXXFLAGS=" -g -O3 -march=native -Wall -fno-strict-aliasing -fopenmp"

AC_OUTPUT([Makefile])
