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

#include <boost/simd/sdk/simd/pack.hpp>
#include <boost/simd/include/functions/plus.hpp>
#include <boost/simd/include/functions/multiplies.hpp>
#include <boost/simd/memory/functions/load.hpp>
#include <boost/simd/memory/functions/aligned_store.hpp>

#include <boost/simd/arithmetic/functions/fma.hpp>

#include <boost/simd/include/functions/load.hpp>
#include <boost/simd/include/functions/store.hpp>

#include "boostSimd.h"

inline size_t getIndex( size_t x, size_t y, size_t width )
{
    return( y * width + x );
}

void printMatrix( const std::string& name, const t_dataVector& matrix, size_t width, size_t height )
{
    //std::cout << name << std::endl << "--------------------------------------------------------------" << std::endl;
    //std::cout << std::fixed << std::setprecision(2) << std::right;
    //for( size_t y = 0; y < height; ++y )
    //{
    //    for( size_t x = 0; x < width; ++x )
    //    {
    //        std::cout.width(13);
    //        std::cout << matrix[ getIndex(x,y,width) ];
    //    }
    //    std::cout << std::endl;
    //}
    //std::cout << "--------------------------------------------------------------" << std::endl;
}

void setupMatrix( t_dataVector& matrix )
{
    for( size_t i = 0; i < matrix.size(); ++i )
    {
        matrix[ i ] = static_cast<t_dataType>( rand() );
    }

    matrix.reserve( matrix.size() + 16*sizeof(t_dataType) );
}

void simpleTransform( t_dataVector& matrix, t_dataVector& factor )
{
    size_t width = factor.size();
    for( size_t line = 0; line < width - 1; ++line )
    {
        for( size_t y = line + 1; y < width; ++y )
        {
            t_dataType scale = matrix[ getIndex( line, y, width ) ] / matrix[ getIndex( line, line, width ) ];

            for( size_t x = line; x < width; ++x )
            {
                matrix[ getIndex( x, y, width ) ] -= scale * matrix[ getIndex( x, line, width ) ];
            }

            factor[ y ] -= scale * factor[ line ];
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

            size_t x = line;
            while( x < endWidth )
            {
                matrix[ getIndex( x, y, width ) ] -= scale * matrix[ getIndex( x, line, width ) ]; ++x;
                matrix[ getIndex( x, y, width ) ] -= scale * matrix[ getIndex( x, line, width ) ]; ++x;
                matrix[ getIndex( x, y, width ) ] -= scale * matrix[ getIndex( x, line, width ) ]; ++x;
                matrix[ getIndex( x, y, width ) ] -= scale * matrix[ getIndex( x, line, width ) ]; ++x;
            }

            while( x < width )
            {
                matrix[ getIndex( x, y, width ) ] -= scale * matrix[ getIndex( x, line, width ) ]; ++x;
            }

            factor[ y ] -= scale * factor[ line ];
        }
    }
}

void openMPTransform( t_dataVector& matrix, t_dataVector& factor )
{
    int width = static_cast<int>(factor.size());
    for( int line = 0; line < width - 1; ++line )
    {
        #pragma omp parallel for schedule(static)
        for( int y = line + 1; y < width; ++y )
        {
            t_dataType scale = matrix[ getIndex( line, y, width ) ] / matrix[ getIndex( line, line, width ) ];

            for( int x = line; x < width; ++x )
            {
                matrix[ getIndex( x, y, width ) ] -= scale * matrix[ getIndex( x, line, width ) ];
            }

            factor[ y ] -= scale * factor[ line ];
        }
    }
}

void unrolledOpenMPTransform( t_dataVector& matrix, t_dataVector& factor )
{
    int width = static_cast<int>(factor.size());
    for( int line = 0; line < width - 1; ++line )
    {
        #pragma omp parallel for schedule(static)
        for( int y = line + 1; y < width; ++y )
        {
            int endWidth = line + ((width - line) & ~(3));

            t_dataType scale = matrix[ getIndex( line, y, width ) ] / matrix[ getIndex( line, line, width ) ];

            int x = line;
            while( x < endWidth )
            {
                matrix[ getIndex( x, y, width ) ] -= scale * matrix[ getIndex( x, line, width ) ]; ++x;
                matrix[ getIndex( x, y, width ) ] -= scale * matrix[ getIndex( x, line, width ) ]; ++x;
                matrix[ getIndex( x, y, width ) ] -= scale * matrix[ getIndex( x, line, width ) ]; ++x;
                matrix[ getIndex( x, y, width ) ] -= scale * matrix[ getIndex( x, line, width ) ]; ++x;
            }

            while( x < width )
            {
                matrix[ getIndex( x, y, width ) ] -= scale * matrix[ getIndex( x, line, width ) ]; ++x;
            }

            factor[ y ] -= scale * factor[ line ];
        }
    }
}

void simdTransform( t_dataVector& matrix, t_dataVector& factor )
{
    typedef boost::simd::pack<t_dataType> t_pack;

    size_t width = factor.size();
    for( size_t line = 0; line < width - 1; ++line )
    {
        for( size_t y = line + 1; y < width; ++y )
        {
            t_pack scale( -matrix[ getIndex( line, y, width ) ] / matrix[ getIndex( line, line, width ) ] );

            size_t normLine = line & ~(static_cast<size_t>(t_pack::static_size - 1));

            for( size_t x = normLine; x < width; x += t_pack::static_size )
            {
                boost::simd::store( boost::simd::fma( scale, boost::simd::aligned_load<t_pack>( matrix.data(), getIndex( x, line, width ) ),
                                                      boost::simd::aligned_load<t_pack>( matrix.data(), getIndex( x, y, width ) ) ), 
                                    matrix.data(), getIndex( x, y, width ) );
            }

            factor[ y ] += scale[0] * factor[ line ];
        }
    }
}

void unrolledSimdTransform( t_dataVector& matrix, t_dataVector& factor )
{
    typedef boost::simd::pack<t_dataType> t_pack;

    size_t width = factor.size();
    for( size_t line = 0; line < width - 1; ++line )
    {
        size_t normLine = line & ~(static_cast<size_t>(t_pack::static_size - 1));
        size_t endWidth = normLine+ ((width - normLine) & ~(4*t_pack::static_size - 1));

        for( size_t y = line + 1; y < width; ++y )
        {
            t_pack scale( -matrix[ getIndex( line, y, width ) ] / matrix[ getIndex( line, line, width ) ] );

            size_t x = normLine;
            while( x < endWidth )
            {
                boost::simd::store( boost::simd::fma( scale, boost::simd::aligned_load<t_pack>( matrix.data(), getIndex( x, line, width ) ),
                                                      boost::simd::aligned_load<t_pack>( matrix.data(), getIndex( x, y, width ) ) ), 
                                    matrix.data(), getIndex( x, y, width ) );
                x += t_pack::static_size;
                boost::simd::store( boost::simd::fma( scale, boost::simd::aligned_load<t_pack>( matrix.data(), getIndex( x, line, width ) ),
                                                      boost::simd::aligned_load<t_pack>( matrix.data(), getIndex( x, y, width ) ) ), 
                                    matrix.data(), getIndex( x, y, width ) );
                x += t_pack::static_size;
                boost::simd::store( boost::simd::fma( scale, boost::simd::aligned_load<t_pack>( matrix.data(), getIndex( x, line, width ) ),
                                                      boost::simd::aligned_load<t_pack>( matrix.data(), getIndex( x, y, width ) ) ), 
                                    matrix.data(), getIndex( x, y, width ) );
                x += t_pack::static_size;
                boost::simd::store( boost::simd::fma( scale, boost::simd::aligned_load<t_pack>( matrix.data(), getIndex( x, line, width ) ),
                                                      boost::simd::aligned_load<t_pack>( matrix.data(), getIndex( x, y, width ) ) ), 
                                    matrix.data(), getIndex( x, y, width ) );
                x += t_pack::static_size;
            }

            while( x < width )
            {
                boost::simd::store( boost::simd::fma( scale, boost::simd::aligned_load<t_pack>( matrix.data(), getIndex( x, line, width ) ),
                                                      boost::simd::aligned_load<t_pack>( matrix.data(), getIndex( x, y, width ) ) ), 
                                    matrix.data(), getIndex( x, y, width ) );
                x += t_pack::static_size;
            }

            factor[ y ] += scale[0] * factor[ line ];
        }
    }
}

void simdOpenMPTransform( t_dataVector& matrix, t_dataVector& factor )
{
    typedef boost::simd::pack<t_dataType> t_pack;

    int width = static_cast<int>(factor.size());
    for( int line = 0; line < width - 1; ++line )
    {
        #pragma omp parallel for schedule(static)
        for( int y = line + 1; y < width; ++y )
        {
            t_pack scale( -matrix[ getIndex( line, y, width ) ] / matrix[ getIndex( line, line, width ) ] );

            int normLine = line & ~(static_cast<int>(t_pack::static_size - 1));

            for( int x = normLine; x < width; x += t_pack::static_size )
            {
                boost::simd::store( boost::simd::fma( scale, boost::simd::aligned_load<t_pack>( matrix.data(), getIndex( x, line, width ) ),
                                                      boost::simd::aligned_load<t_pack>( matrix.data(), getIndex( x, y, width ) ) ), 
                                    matrix.data(), getIndex( x, y, width ) );
            }

            factor[ y ] += scale[0] * factor[ line ];
        }
    }
}

void unrolledSimdOpenMPTransform( t_dataVector& matrix, t_dataVector& factor )
{
    typedef boost::simd::pack<t_dataType> t_pack;

    int width = static_cast<int>(factor.size());
    for( int line = 0; line < width - 1; ++line )
    {
        int normLine = line & ~(4*t_pack::static_size - 1);
        int endWidth = (width - normLine) & ~(4*t_pack::static_size - 1);

        #pragma omp parallel for schedule(static)
        for( int y = line + 1; y < width; ++y )
        {
            t_pack scale( -matrix[ getIndex( line, y, width ) ] / matrix[ getIndex( line, line, width ) ] );

            int x = normLine;
            while( x < endWidth )
            {
                boost::simd::store( boost::simd::fma( scale, boost::simd::aligned_load<t_pack>( matrix.data(), getIndex( x, line, width ) ),
                                                      boost::simd::aligned_load<t_pack>( matrix.data(), getIndex( x, y, width ) ) ), 
                                    matrix.data(), getIndex( x, y, width ) );
                x += t_pack::static_size;
                boost::simd::store( boost::simd::fma( scale, boost::simd::aligned_load<t_pack>( matrix.data(), getIndex( x, line, width ) ),
                                                      boost::simd::aligned_load<t_pack>( matrix.data(), getIndex( x, y, width ) ) ), 
                                    matrix.data(), getIndex( x, y, width ) );
                x += t_pack::static_size;
                boost::simd::store( boost::simd::fma( scale, boost::simd::aligned_load<t_pack>( matrix.data(), getIndex( x, line, width ) ),
                                                      boost::simd::aligned_load<t_pack>( matrix.data(), getIndex( x, y, width ) ) ), 
                                    matrix.data(), getIndex( x, y, width ) );
                x += t_pack::static_size;
                boost::simd::store( boost::simd::fma( scale, boost::simd::aligned_load<t_pack>( matrix.data(), getIndex( x, line, width ) ),
                                                      boost::simd::aligned_load<t_pack>( matrix.data(), getIndex( x, y, width ) ) ), 
                                    matrix.data(), getIndex( x, y, width ) );
                x += t_pack::static_size;
            }

            while( x < width )
            {
                boost::simd::store( boost::simd::fma( scale, boost::simd::aligned_load<t_pack>( matrix.data(), getIndex( x, line, width ) ),
                                                      boost::simd::aligned_load<t_pack>( matrix.data(), getIndex( x, y, width ) ) ), 
                                    matrix.data(), getIndex( x, y, width ) );
                x += t_pack::static_size;
            }

            factor[ y ] += scale[0] * factor[ line ];
        }
    }
}

