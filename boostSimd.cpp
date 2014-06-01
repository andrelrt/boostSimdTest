/*
 * Copyright (c) 2014 Andr� Tupinamb� (andrelrt@gmail.com)
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
#include <time.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

#include <xmmintrin.h>

#include <boost/simd/sdk/simd/pack.hpp>
#include <boost/simd/include/functions/plus.hpp>
#include <boost/simd/include/functions/multiplies.hpp>

#include <boost/simd/memory/functions/load.hpp>
#include <boost/simd/memory/functions/aligned_store.hpp>
#include <boost/simd/memory/functions/stream.hpp>
#include <boost/simd/memory/prefetch.hpp>

#include <boost/simd/arithmetic/functions/fma.hpp>

#include <boost/simd/include/functions/load.hpp>
#include <boost/simd/include/functions/store.hpp>

#include "boostSimd.h"

inline size_t getIndex( size_t x, size_t y, size_t width )
{
	return( y * width + x );
}

void simpleTransform( t_dataVector& matrix, t_dataVector& factor )
{
	size_t width = factor.size();
	for( size_t line = 0; line < width - 1; ++line )
	{
		for( size_t y = line + 1; y < width; ++y )
		{
			t_dataType scale = matrix[ getIndex( line, y, width ) ] / matrix[ getIndex( line, line, width ) ];
            factor[ y ] -= scale * factor[ line ];

            t_dataType* pBase = &( matrix[ getIndex( line, line, width ) ] );
            t_dataType* pLine = &( matrix[ getIndex( line, y, width ) ] );
			for( size_t x = line; x < width; ++x )
			{
				*pLine -= scale * *pBase;
                ++pLine;
                ++pBase;
			}
		}
	}
}

void unrolledTransform( t_dataVector& matrix, t_dataVector& factor )
{
	size_t width = factor.size();
	for( size_t line = 0; line < width - 1; ++line )
	{
		size_t endWidth =  line + ((width - line) & ~(3));

		for( size_t y = line + 1; y < width; ++y )
		{
			t_dataType scale = matrix[ getIndex( line, y, width ) ] / matrix[ getIndex( line, line, width ) ];
            factor[ y ] -= scale * factor[ line ];

			size_t x = line;
            t_dataType* pBase = &( matrix[ getIndex( line, line, width ) ] );
            t_dataType* pLine = &( matrix[ getIndex( line, y, width ) ] );
			while( x < endWidth )
			{
				*pLine -= scale * *pBase; ++pLine; ++pBase;
				*pLine -= scale * *pBase; ++pLine; ++pBase;
				*pLine -= scale * *pBase; ++pLine; ++pBase;
				*pLine -= scale * *pBase; ++pLine; ++pBase;

                x += 4;
			}

			while( x < width )
			{
				*pLine -= scale * *pBase; ++pLine; ++pBase;
				++x;
			}
		}
	}
}

void openMPTransform( t_dataVector& matrix, t_dataVector& factor )
{
	int width = static_cast<int>(factor.size());
	for( int line = 0; line < width - 1; ++line )
	{
		#pragma omp parallel for
		for( int y = line + 1; y < width; ++y )
		{
			t_dataType scale = matrix[ getIndex( line, y, width ) ] / matrix[ getIndex( line, line, width ) ];
            factor[ y ] -= scale * factor[ line ];

            t_dataType* pBase = &( matrix[ getIndex( line, line, width ) ] );
            t_dataType* pLine = &( matrix[ getIndex( line, y, width ) ] );
			for( int x = line; x < width; ++x )
			{
				*pLine -= scale * *pBase;
                ++pLine;
                ++pBase;
			}
		}
	}
}

void unrolledOpenMPTransform( t_dataVector& matrix, t_dataVector& factor )
{
	int width = static_cast<int>(factor.size());
	for( int line = 0; line < width - 1; ++line )
	{
		#pragma omp parallel for
		for( int y = line + 1; y < width; ++y )
		{
			int endWidth = line + ((width - line) & ~(3));

			t_dataType scale = matrix[ getIndex( line, y, width ) ] / matrix[ getIndex( line, line, width ) ];
            factor[ y ] -= scale * factor[ line ];

			int x = line;
            t_dataType* pBase = &( matrix[ getIndex( line, line, width ) ] );
            t_dataType* pLine = &( matrix[ getIndex( line, y, width ) ] );
			while( x < endWidth )
			{
				*pLine -= scale * *pBase; ++pLine; ++pBase;
				*pLine -= scale * *pBase; ++pLine; ++pBase;
				*pLine -= scale * *pBase; ++pLine; ++pBase;
				*pLine -= scale * *pBase; ++pLine; ++pBase;

                x += 4;
			}

			while( x < width )
			{
				*pLine -= scale * *pBase; ++pLine; ++pBase;
				++x;
			}
		}
	}
}

void simdTransform( t_dataVector& matrix, t_dataVector& factor )
{
	typedef boost::simd::pack<t_dataType> t_pack;

	size_t width = factor.size();
    t_pack* packMatrix = reinterpret_cast<t_pack*>( matrix.data() );
	for( size_t line = 0; line < width - 1; ++line )
	{
		for( size_t y = line + 1; y < width; ++y )
		{
            t_dataType scale = matrix[ getIndex( line, y, width ) ] / matrix[ getIndex( line, line, width ) ];
            factor[ y ] -= scale * factor[ line ];

			size_t normLine = line & ~(static_cast<size_t>(t_pack::static_size - 1));

            t_pack* packLine = &( packMatrix[ getIndex( normLine, y, width ) / t_pack::static_size ] );
            t_pack* packBase = &( packMatrix[ getIndex( normLine, line, width ) / t_pack::static_size ] );
            register t_pack packScale( -scale );
            for( size_t x = normLine; x < width; x += t_pack::static_size )
			{
                *packLine = boost::simd::fma( packScale, *packBase, *packLine );
                ++packLine; ++packBase;
			}
		}
	}
}

void unrolledSimdTransform( t_dataVector& matrix, t_dataVector& factor )
{
	typedef boost::simd::pack<t_dataType> t_pack;

	size_t width = factor.size();
    t_pack* packMatrix = reinterpret_cast<t_pack*>( matrix.data( ) );
    for( size_t line = 0; line < width - 1; ++line )
	{
		size_t normLine = line & ~(static_cast<size_t>(t_pack::static_size - 1));
		size_t endWidth = normLine + ((width - normLine) & ~(4*t_pack::static_size - 1));

		for( size_t y = line + 1; y < width; ++y )
		{
            t_dataType scale = matrix[ getIndex( line, y, width ) ] / matrix[ getIndex( line, line, width ) ];
            factor[ y ] -= scale * factor[ line ];

			size_t x = normLine;
            t_pack* packLine = &( packMatrix[ getIndex( normLine, y, width ) / t_pack::static_size ] );
            t_pack* packBase = &( packMatrix[ getIndex( normLine, line, width ) / t_pack::static_size ] );
            register t_pack packScale( -scale );

            while( x < endWidth )
			{
                *packLine = boost::simd::fma( packScale, *packBase, *packLine ); ++packLine; ++packBase;
                *packLine = boost::simd::fma( packScale, *packBase, *packLine ); ++packLine; ++packBase;
                *packLine = boost::simd::fma( packScale, *packBase, *packLine ); ++packLine; ++packBase;
                *packLine = boost::simd::fma( packScale, *packBase, *packLine ); ++packLine; ++packBase;

                x += 4*t_pack::static_size;
			}

			while( x < width )
			{
                *packLine = boost::simd::fma( packScale, *packBase, *packLine ); ++packLine; ++packBase;

                x += t_pack::static_size;
			}
		}
	}
}

void simdOpenMPTransform( t_dataVector& matrix, t_dataVector& factor )
{
	typedef boost::simd::pack<t_dataType> t_pack;

	int width = static_cast<int>(factor.size());
    t_pack* packMatrix = reinterpret_cast<t_pack*>( matrix.data( ) );
    for( int line = 0; line < width - 1; ++line )
	{
		#pragma omp parallel for
		for( int y = line + 1; y < width; ++y )
		{
            t_dataType scale = matrix[ getIndex( line, y, width ) ] / matrix[ getIndex( line, line, width ) ];
            factor[ y ] -= scale * factor[ line ];

			int normLine = line & ~(static_cast<int>(t_pack::static_size - 1));

            t_pack* packLine = &( packMatrix[ getIndex( normLine, y, width ) / t_pack::static_size ] );
            t_pack* packBase = &( packMatrix[ getIndex( normLine, line, width ) / t_pack::static_size ] );
            register t_pack packScale( -scale );
            for( size_t x = normLine; x < width; x += t_pack::static_size )
            {
                *packLine = boost::simd::fma( packScale, *packBase, *packLine );
                ++packLine; ++packBase;
            }
        }
	}
}

void unrolledSimdOpenMPTransform( t_dataVector& matrix, t_dataVector& factor )
{
	typedef boost::simd::pack<t_dataType> t_pack;

	int width = static_cast<int>(factor.size());
    t_pack* packMatrix = reinterpret_cast<t_pack*>( matrix.data( ) );
    for( int line = 0; line < width - 1; ++line )
	{
		int normLine = line & ~(4*t_pack::static_size - 1);
		int endWidth = (width - normLine) & ~(4*t_pack::static_size - 1);

		#pragma omp parallel for
		for( int y = line + 1; y < width; ++y )
		{
            t_dataType scale = matrix[ getIndex( line, y, width ) ] / matrix[ getIndex( line, line, width ) ];
            factor[ y ] -= scale * factor[ line ];

			int x = normLine;
            t_pack* packLine = &( packMatrix[ getIndex( normLine, y, width ) / t_pack::static_size ] );
            t_pack* packBase = &( packMatrix[ getIndex( normLine, line, width ) / t_pack::static_size ] );
            register t_pack packScale( -scale );

            while( x < endWidth )
            {
                *packLine = boost::simd::fma( packScale, *packBase, *packLine ); ++packLine; ++packBase;
                *packLine = boost::simd::fma( packScale, *packBase, *packLine ); ++packLine; ++packBase;
                *packLine = boost::simd::fma( packScale, *packBase, *packLine ); ++packLine; ++packBase;
                *packLine = boost::simd::fma( packScale, *packBase, *packLine ); ++packLine; ++packBase;

                x += 4 * t_pack::static_size;
            }

            while( x < width )
            {
                *packLine = boost::simd::fma( packScale, *packBase, *packLine ); ++packLine; ++packBase;

                x += t_pack::static_size;
            }
        }
	}
}

#ifdef BUILD_INTRINSICS_TRANSFORMS
void intrinsicsTransformFloat( t_dataVector& matrix, t_dataVector& factor )
{
	size_t width = factor.size();
	for( size_t line = 0; line < width - 1; ++line )
	{
		size_t normLine = line & ~(3);

		for( size_t y = line + 1; y < width; ++y )
		{
			float scale = matrix[ getIndex( line, y, width ) ] / matrix[ getIndex( line, line, width ) ];
			__m128 xmmScale = _mm_set1_ps( scale );

			factor[ y ] -= scale * factor[ line ];

			for( size_t x = normLine; x < width; x += 4 )
			{
				_mm_store_ps( matrix.data() + getIndex( x, y, width ),
							  _mm_sub_ps( _mm_load_ps( matrix.data() + getIndex( x, y, width ) ),
										  _mm_mul_ps( xmmScale,
													  _mm_load_ps( matrix.data() + getIndex( x, line, width ) ) ) ) );
			}
		}
	}
}

void unrolledIntrinsicsTransformFloat( t_dataVector& matrix, t_dataVector& factor )
{
	size_t width = factor.size();
	for( size_t line = 0; line < width - 1; ++line )
	{
		size_t normLine = line & ~(3);
		size_t endWidth = normLine + ((width - normLine) & ~(15));

		for( size_t y = line + 1; y < width; ++y )
		{
			float scale = matrix[ getIndex( line, y, width ) ] / matrix[ getIndex( line, line, width ) ];
			__m128 xmmScale = _mm_set1_ps( scale );

			factor[ y ] -= scale * factor[ line ];

			size_t x = normLine;
			while( x < endWidth )
			{
				_mm_store_ps( matrix.data() + getIndex( x, y, width ),
							  _mm_sub_ps( _mm_load_ps( matrix.data() + getIndex( x, y, width ) ),
										  _mm_mul_ps( xmmScale,
													  _mm_load_ps( matrix.data() + getIndex( x, line, width ) ) ) ) );
				x += 4;
				_mm_store_ps( matrix.data() + getIndex( x, y, width ),
							  _mm_sub_ps( _mm_load_ps( matrix.data() + getIndex( x, y, width ) ),
										  _mm_mul_ps( xmmScale,
													  _mm_load_ps( matrix.data() + getIndex( x, line, width ) ) ) ) );
				x += 4;
				_mm_store_ps( matrix.data() + getIndex( x, y, width ),
							  _mm_sub_ps( _mm_load_ps( matrix.data() + getIndex( x, y, width ) ),
										  _mm_mul_ps( xmmScale,
													  _mm_load_ps( matrix.data() + getIndex( x, line, width ) ) ) ) );
				x += 4;
				_mm_store_ps( matrix.data() + getIndex( x, y, width ),
							  _mm_sub_ps( _mm_load_ps( matrix.data() + getIndex( x, y, width ) ),
										  _mm_mul_ps( xmmScale,
													  _mm_load_ps( matrix.data() + getIndex( x, line, width ) ) ) ) );
				x += 4;
			}

			while( x < width )
			{
				_mm_store_ps( matrix.data() + getIndex( x, y, width ),
							  _mm_sub_ps( _mm_load_ps( matrix.data() + getIndex( x, y, width ) ),
										  _mm_mul_ps( xmmScale,
													  _mm_load_ps( matrix.data() + getIndex( x, line, width ) ) ) ) );
				x += 4;
			}
		}
	}
}

void intrinsicsOpenMPTransformFloat( t_dataVector& matrix, t_dataVector& factor )
{
	int width = static_cast<int>( factor.size() );
	for( int line = 0; line < width - 1; ++line )
	{
		int normLine = line & ~(3);

		#pragma omp parallel for schedule(static)
		for( int y = line + 1; y < width; ++y )
		{
			float scale = matrix[ getIndex( line, y, width ) ] / matrix[ getIndex( line, line, width ) ];
			__m128 xmmScale = _mm_set1_ps( scale );

			factor[ y ] -= scale * factor[ line ];

			for( int x = normLine; x < width; x += 4 )
			{
				_mm_store_ps( matrix.data() + getIndex( x, y, width ),
							  _mm_sub_ps( _mm_load_ps( matrix.data() + getIndex( x, y, width ) ),
										  _mm_mul_ps( xmmScale,
													  _mm_load_ps( matrix.data() + getIndex( x, line, width ) ) ) ) );
			}
		}
	}
}

void unrolledIntrinsicsOpenMPTransformFloat( t_dataVector& matrix, t_dataVector& factor )
{
	int width = static_cast<int>( factor.size() );
	for( int line = 0; line < width - 1; ++line )
	{
		int normLine = line & ~(3);
		int endWidth = normLine + ((width - normLine) & ~(15));

		#pragma omp parallel for schedule(static)
		for( int y = line + 1; y < width; ++y )
		{
			float scale = matrix[ getIndex( line, y, width ) ] / matrix[ getIndex( line, line, width ) ];
			__m128 xmmScale = _mm_set1_ps( scale );

			factor[ y ] -= scale * factor[ line ];

			int x = normLine;
			while( x < endWidth )
			{
				_mm_store_ps( matrix.data() + getIndex( x, y, width ),
							  _mm_sub_ps( _mm_load_ps( matrix.data() + getIndex( x, y, width ) ),
										  _mm_mul_ps( xmmScale,
													  _mm_load_ps( matrix.data() + getIndex( x, line, width ) ) ) ) );
				x += 4;
				_mm_store_ps( matrix.data() + getIndex( x, y, width ),
							  _mm_sub_ps( _mm_load_ps( matrix.data() + getIndex( x, y, width ) ),
										  _mm_mul_ps( xmmScale,
													  _mm_load_ps( matrix.data() + getIndex( x, line, width ) ) ) ) );
				x += 4;
				_mm_store_ps( matrix.data() + getIndex( x, y, width ),
							  _mm_sub_ps( _mm_load_ps( matrix.data() + getIndex( x, y, width ) ),
										  _mm_mul_ps( xmmScale,
													  _mm_load_ps( matrix.data() + getIndex( x, line, width ) ) ) ) );
				x += 4;
				_mm_store_ps( matrix.data() + getIndex( x, y, width ),
							  _mm_sub_ps( _mm_load_ps( matrix.data() + getIndex( x, y, width ) ),
										  _mm_mul_ps( xmmScale,
													  _mm_load_ps( matrix.data() + getIndex( x, line, width ) ) ) ) );
				x += 4;
			}

			while( x < width )
			{
				_mm_store_ps( matrix.data() + getIndex( x, y, width ),
							  _mm_sub_ps( _mm_load_ps( matrix.data() + getIndex( x, y, width ) ),
										  _mm_mul_ps( xmmScale,
													  _mm_load_ps( matrix.data() + getIndex( x, line, width ) ) ) ) );
				x += 4;
			}
		}
	}
}
#endif // BUILD_INTRINSICS_TRANSFORMS

