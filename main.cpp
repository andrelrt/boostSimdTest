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
#include "boostSimd.h"
#include <iostream>
#include <iomanip>
#include <boost/timer/timer.hpp>

void setupMatrix( t_dataVector& matrix );
void printMatrix( const std::string& name, const t_dataVector& matrix, size_t width, size_t height );

int main()
{
    // Disable denormals
    // Requires #include <xmmintrin.h>
    // Requires #include <pmmintrin.h>
    _MM_SET_FLUSH_ZERO_MODE( _MM_FLUSH_ZERO_ON );
    _MM_SET_DENORMALS_ZERO_MODE( _MM_DENORMALS_ZERO_ON );

    size_t width = 768; // Always multiples of 16
    size_t height = width; // Always a square matrix 

    size_t loopCount = 200;

    struct Executions
    {
        std::string name_;
        void (*transform_)( t_dataVector& matrix, t_dataVector& factor );
    };

    Executions exec[] = {
//    { "Base",                       &simpleTransform },
//    { "Unrolled",                   &unrolledTransform },
//    { "OpenMP",                     &openMPTransform },
//    { "OpenMP unrolled",            &unrolledOpenMPTransform },

//    { "Vectorized", &vectorizedTransform },
//    { "Vectorized OpenMP", &vectorizedOpenMPTransform },

    { "Boost.SIMD", &simdTransform },
    { "Boost.SIMD with ranges", &simdTransform2 },
    { "Boost.SIMD with transform", &simdTransform3 },
//    { "Boost.SIMD unrolled",        &unrolledSimdTransform },
//    { "Boost.SIMD OpenMP",          &simdOpenMPTransform },
 //   { "Boost.SIMD OpenMP unrolled", &unrolledSimdOpenMPTransform },

#ifdef BUILD_INTRINSICS_TRANSFORMS
//    { "Intrinsics Float",           &intrinsicsTransformFloat },
//    { "Intrinsics Float unrolled",  &unrolledIntrinsicsTransformFloat },
//    { "Intrinsics Float OpenMP",    &intrinsicsOpenMPTransformFloat },
//    { "Intrinsics Float OpenMP unrolled", &unrolledIntrinsicsOpenMPTransformFloat },
#endif // BUILD_INTRINSICS_TRANSFORMS

    { "", NULL } };

    t_dataVector baseMatrix( width * height );
    t_dataVector baseFactor( width );

    boost::timer::cpu_timer timer;

    srand( time(NULL) ); // To debug put a constant here
    setupMatrix( baseMatrix );
    setupMatrix( baseFactor );

    printMatrix( "Matrix", baseMatrix, width, height );
    printMatrix( "Factors", baseFactor, width, 1 );

    size_t index = 0;
    while( exec[index].transform_ )
    {
        t_dataVector matrix( baseMatrix );
        t_dataVector factor( baseFactor );

        // Warmup (fill cache, create OpenMP threads, etc).
        exec[index].transform_( matrix, factor );

        timer.start();
        for(size_t i = 0; i < loopCount; ++i )
        {
            matrix = baseMatrix;
            factor = baseFactor;
            exec[index].transform_( matrix, factor );
        }
        timer.stop();
        printMatrix( "Matrix", matrix, width, height );
        printMatrix( "Factors", factor, width, 1 );

        double seconds = static_cast<double>(timer.elapsed().wall) / 1000000000.0;
        double matrixPerSecond = loopCount / seconds;
        double kflops = static_cast<double>(width*(width + 2) + 2 * width)/ 1000.0 / seconds ;

        std::cout << exec[index].name_ << " time: " 
                  << timer.format( boost::timer::default_places, "%ws wall, %us user + %ss system = %ts CPU (%p%)" )
                  << " - " << std::fixed << std::setprecision(6) << matrixPerSecond << " matrix/s"
                  << " - " << std::fixed << std::setprecision(2) << kflops << " kflops"
                  << std::endl << std::endl;

        ++index;
    }
    return 0;
}

void setupMatrix( t_dataVector& matrix )
{
    std::generate( matrix.begin( ), matrix.end( ), &rand );

    matrix.reserve( matrix.size( ) + 16 * sizeof( t_dataType ) );
}

void printMatrix( const std::string& name, const t_dataVector& matrix, size_t width, size_t height )
{
    //	std::cout << name << std::endl << "--------------------------------------------------------------" << std::endl;
    //	std::cout << std::fixed << std::setprecision(2) << std::right;
    //	for( size_t y = 0; y < height; ++y )
    //	{
    //	    for( size_t x = 0; x < width; ++x )
    //	    {
    //	        std::cout.width(13);
    //	        std::cout << matrix[ getIndex(x,y,width) ];
    //	    }
    //	    std::cout << std::endl;
    //	}
    //	std::cout << "--------------------------------------------------------------" << std::endl;
}

