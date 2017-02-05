# - Find Boost.SIMD
# Find the native Boost.SIMD includes
#
#  BOOST_SIMD_INCLUDE_DIR    - where to find boost/simd/pack.hpp
#  BOOST_SIMD_FOUND          - True if Boost.SIMD found.

if (BOOST_SIMD_INCLUDE_DIR)
  # Already in cache, be silent
  set (BOOST_SIMD_FIND_QUIETLY TRUE)
endif (BOOST_SIMD_INCLUDE_DIR)

if (NOT "$ENV{BOOST_SIMD_ROOT}" STREQUAL "")
  set (BOOST_SIMD_HINT $ENV{BOOST_SIMD_ROOT}/include)
endif()

find_path (BOOST_SIMD_INCLUDE_DIR boost/simd/pack.hpp HINTS ${BOOST_SIMD_HINT})

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (Boost.SIMD DEFAULT_MSG BOOST_SIMD_INCLUDE_DIR)

mark_as_advanced (BOOST_SIMD_INCLUDE_DIR)
