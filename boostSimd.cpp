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
#include <time.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

#include <xmmintrin.h>

#include <boost/simd/pack.hpp>
#include <boost/simd/function/plus.hpp>
#include <boost/simd/function/multiplies.hpp>

#include <boost/simd/function/load.hpp>
#include <boost/simd/function/aligned_store.hpp>
#include <boost/simd/function/stream.hpp>
//#include <boost/simd/prefetch.hpp>

#include <boost/simd/function/fma.hpp>

#include <boost/simd/function/load.hpp>
#include <boost/simd/function/store.hpp>
#include <boost/simd/algorithm/transform.hpp>

#include <boost/simd/range/aligned_input_range.hpp>
#include <boost/simd/range/aligned_output_range.hpp>

#include "boostSimd.h"

namespace bs = boost::simd;

inline size_t getIndex( size_t x, size_t y, size_t width )
{
	return( y * width + x );
}

void simpleTransform( t_dataVector& matrix, t_dataVector& factor )
{
    size_t width = factor.size( );
    for( size_t line = 0; line < width - 1; ++line )
    {
        for( size_t y = line + 1; y < width; ++y )
        {
             t_dataType scale = matrix[ getIndex( line, y, width ) ] / matrix[ getIndex( line, line, width ) ];
            factor[ y ] -= scale * factor[ line ];

            for( size_t x = line; x < width; ++x )
            {
                matrix[ getIndex( x, y, width ) ] -= scale * matrix[ getIndex( x, line, width ) ];
            }
        }
    }
}

void vectorizedTransform( t_dataVector& matrix, t_dataVector& factor )
{
    size_t width = factor.size( );
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
                *pLine++ -= scale * *pBase++;
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
                *pLine++ -= scale * *pBase++;
                *pLine++ -= scale * *pBase++;
                *pLine++ -= scale * *pBase++;
                *pLine++ -= scale * *pBase++;

                x += 4;
			}

			while( x < width )
			{
                *pLine++ -= scale * *pBase++;
                ++x;
			}
		}
	}
}

#ifdef _OPENMP
void openMPTransform(t_dataVector& matrix, t_dataVector& factor)
{
    int width = static_cast<int>(factor.size());
    for (int line = 0; line < width - 1; ++line)
    {
#pragma omp parallel for
        for (int y = line + 1; y < width; ++y)
        {
            t_dataType scale = matrix[getIndex(line, y, width)] / matrix[getIndex(line, line, width)];
            factor[y] -= scale * factor[line];

            for (size_t x = line; x < width; ++x)
            {
                matrix[getIndex(x, y, width)] -= scale * matrix[getIndex(x, line, width)];
            }
        }
    }
}

void vectorizedOpenMPTransform(t_dataVector& matrix, t_dataVector& factor)
{
    int width = static_cast<int>(factor.size());
    for (int line = 0; line < width - 1; ++line)
    {
#pragma omp parallel for
        for (int y = line + 1; y < width; ++y)
        {
            t_dataType scale = matrix[getIndex(line, y, width)] / matrix[getIndex(line, line, width)];
            factor[y] -= scale * factor[line];

            t_dataType* pBase = &(matrix[getIndex(line, line, width)]);
            t_dataType* pLine = &(matrix[getIndex(line, y, width)]);
            for (int x = line; x < width; ++x)
            {
                *pLine++ -= scale * *pBase++;
            }
        }
    }
}

void unrolledOpenMPTransform(t_dataVector& matrix, t_dataVector& factor)
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
                *pLine++ -= scale * *pBase++;
                *pLine++ -= scale * *pBase++;
                *pLine++ -= scale * *pBase++;
                *pLine++ -= scale * *pBase++;

                x += 4;
			}

			while( x < width )
			{
                *pLine++ -= scale * *pBase++;
                ++x;
			}
		}
	}
}
#endif // _OPENMP

void simdTransform( t_dataVector& matrix, t_dataVector& factor )
{
	using t_pack = bs::pack<t_dataType>;

	size_t width = factor.size();
    t_pack* packMatrix = reinterpret_cast<t_pack*>( matrix.data() );
	for( size_t line = 0; line < width - 1; ++line )
	{
        size_t normLine = line & ~(static_cast<size_t>(t_pack::static_size - 1));
		for( size_t y = line + 1; y < width; ++y )
		{
            t_dataType scale = matrix[ getIndex( line, y, width ) ] / matrix[ getIndex( line, line, width ) ];
            factor[ y ] = bs::fma( -scale, factor[ line ], factor[ y ] );

            t_pack* packLine = &( packMatrix[ getIndex( normLine, y, width ) / t_pack::static_size ] );
            t_pack* packBase = &( packMatrix[ getIndex( normLine, line, width ) / t_pack::static_size ] );
            t_pack packScale( -scale );
            for( size_t x = normLine; x < width; x += t_pack::static_size )
			{
                *packLine = bs::fma( packScale, *packBase++, *packLine );
                packLine++;
			}
		}
	}
}

struct myfma
{
    t_dataType scale;
    template<typename T> 
    T operator()(T const& a, T const& b) const
    {
        return bs::fma( scale, b, a );
    }
};

void simdTransform2( t_dataVector& matrix, t_dataVector& factor )
{
	using t_pack = bs::pack<t_dataType>;

	size_t width = factor.size();
	for( size_t line = 0; line < width - 1; ++line )
	{
        size_t normLine = line & ~(static_cast<size_t>(t_pack::static_size - 1));
        t_dataType* pBase = &( matrix[ getIndex( normLine, line, width ) ] );
        auto baseRange = bs::aligned_input_range( pBase, pBase + (width - normLine) );
        for( size_t y = line + 1; y < width; ++y )
		{
            t_dataType scale = matrix[ getIndex( line, y, width ) ] / matrix[ getIndex( line, line, width ) ];
            factor[ y ] = bs::fma( -scale, factor[ line ], factor[ y ] );

            t_dataType* pLine = &( matrix[ getIndex( normLine, y, width ) ] );
            auto lineIt = std::begin( bs::aligned_input_range( pLine, pLine + (width - normLine) ) );
            auto outIt = std::begin( bs::aligned_output_range( pLine, pLine + (width - normLine) ) );
            t_pack packScale( -scale );
            for( auto&& val : baseRange ) 
            {
                *outIt = bs::fma( packScale, val, *lineIt ); ++lineIt; ++outIt;
            }
		}
	}
}

void simdTransform3( t_dataVector& matrix, t_dataVector& factor )
{
    using t_pack = bs::pack<t_dataType>;
    size_t width = factor.size( );
    for( size_t line = 0; line < width - 1; ++line )
    {
        for( size_t y = line + 1; y < width; ++y )
        {
            t_dataType scale = matrix[ getIndex( line, y, width ) ] / matrix[ getIndex( line, line, width ) ];
            factor[ y ] = bs::fma( -scale, factor[ line ], factor[ y ] );

			size_t normLine = line & ~(static_cast<size_t>(t_pack::static_size - 1));
            t_dataType* pBase = &( matrix[ getIndex( normLine, line, width ) ] );
            t_dataType* pLine = &( matrix[ getIndex( normLine, y, width ) ] );

            myfma f;
            f.scale = -scale;
            bs::transform( pLine, pLine + (width - normLine), pBase, pLine, f );
        }
    }
}

void unrolledSimdTransform( t_dataVector& matrix, t_dataVector& factor )
{
	using t_pack = bs::pack<t_dataType>;

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
            t_pack packScale( -scale );

            while( x < endWidth )
			{
                *packLine = bs::fma( packScale, *packBase, *packLine ); ++packLine; ++packBase;
                *packLine = bs::fma( packScale, *packBase, *packLine ); ++packLine; ++packBase;
                *packLine = bs::fma( packScale, *packBase, *packLine ); ++packLine; ++packBase;
                *packLine = bs::fma( packScale, *packBase, *packLine ); ++packLine; ++packBase;

                x += 4*t_pack::static_size;
			}

			while( x < width )
			{
                *packLine = bs::fma( packScale, *packBase, *packLine ); ++packLine; ++packBase;

                x += t_pack::static_size;
			}
		}
	}
}

#ifdef _OPENMP
void simdOpenMPTransform( t_dataVector& matrix, t_dataVector& factor )
{
	using t_pack = bs::pack<t_dataType>;

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
            t_pack packScale( -scale );
            for( size_t x = normLine; x < width; x += t_pack::static_size )
            {
                *packLine = bs::fma(packScale, *packBase++, *packLine); ++packLine;
            }
        }
	}
}

void unrolledSimdOpenMPTransform( t_dataVector& matrix, t_dataVector& factor )
{
	using t_pack = bs::pack<t_dataType>;

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
            t_pack packScale( -scale );

            while( x < endWidth )
            {
                *packLine = bs::fma(packScale, *packBase++, *packLine); ++packLine;
                *packLine = bs::fma(packScale, *packBase++, *packLine); ++packLine;
                *packLine = bs::fma(packScale, *packBase++, *packLine); ++packLine;
                *packLine = bs::fma(packScale, *packBase++, *packLine); ++packLine;
                x += 4 * t_pack::static_size;
            }

            while( x < width )
            {
                *packLine = bs::fma(packScale, *packBase++, *packLine); ++packLine;

                x += t_pack::static_size;
            }
        }
	}
}
#endif // _OPENMP

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

#ifdef _OPENMP
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
#endif // _OPENMP
#endif // BUILD_INTRINSICS_TRANSFORMS

