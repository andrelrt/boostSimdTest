/*
 * Copyright (c) 2014 André Tupinambá (andrelrt@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
//-----------------------------------------------------------------------------
#ifndef __BOOST_SIMD_TEST__
#define __BOOST_SIMD_TEST__

#include <vector>
#include <boost/simd/memory/allocator.hpp>

#define BUILD_INTRINSICS_TRANSFORMS 1
using t_dataType = float;
using t_dataVector = std::vector<t_dataType, boost::simd::allocator<t_dataType>>;

void simpleTransform( t_dataVector& matrix, t_dataVector& factor );
void unrolledTransform( t_dataVector& matrix, t_dataVector& factor );
void vectorizedTransform(t_dataVector& matrix, t_dataVector& factor);
void simdTransform( t_dataVector& matrix, t_dataVector& factor );
void simdTransform2( t_dataVector& matrix, t_dataVector& factor );
void simdTransform3( t_dataVector& matrix, t_dataVector& factor );
void unrolledSimdTransform( t_dataVector& matrix, t_dataVector& factor );

#ifdef BUILD_INTRINSICS_TRANSFORMS
void intrinsicsTransformFloat( t_dataVector& matrix, t_dataVector& factor );
void unrolledIntrinsicsTransformFloat( t_dataVector& matrix, t_dataVector& factor );
#endif // BUILD_INTRINSICS_TRANSFORMS

#ifdef _OPENMP
void openMPTransform( t_dataVector& matrix, t_dataVector& factor );
void unrolledOpenMPTransform( t_dataVector& matrix, t_dataVector& factor );
void vectorizedOpenMPTransform(t_dataVector& matrix, t_dataVector& factor);
void simdOpenMPTransform( t_dataVector& matrix, t_dataVector& factor );
void unrolledSimdOpenMPTransform( t_dataVector& matrix, t_dataVector& factor );
#ifdef BUILD_INTRINSICS_TRANSFORMS
void intrinsicsOpenMPTransformFloat( t_dataVector& matrix, t_dataVector& factor );
void unrolledIntrinsicsOpenMPTransformFloat( t_dataVector& matrix, t_dataVector& factor );
#endif // BUILD_INTRINSICS_TRANSFORMS
#endif // _OPENMP

#endif // __BOOST_SIMD_TEST__
