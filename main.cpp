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
#include <boost/timer/timer.hpp>

int main()
{
    size_t width = 500; // Coloque sempre números múltiplos de 4
    size_t height = width; // Testei somente com matrizes quadradas

    struct Executions
    {
        std::string name_;
        void (*transform_)( t_dataVector& matrix, t_dataVector& factor );
    };

    Executions exec[] = {
    { "Base",                       &simpleTransform },
    { "Unrolled",                   &unrolledTransform },
    { "OpenMP",                     &openMPTransform },
    { "OpenMP unrolled",            &unrolledOpenMPTransform },
    { "Boost.SIMD",                 &simdTransform },
    { "Boost.SIMD unrolled",        &unrolledSimdTransform },
    { "Boost.SIMD OpenMP",          &simdOpenMPTransform },
    { "Boost.SIMD OpenMP unrolled", &unrolledSimdOpenMPTransform },
    { "", NULL } };

    t_dataVector baseMatrix( width * height );
    t_dataVector baseFactor( width );

    boost::timer::cpu_timer timer;

    srand( time(NULL) ); // Para depurar é melhor colocar uma constante aqui, vai ser sempre a mesma matriz
    setupMatrix( baseMatrix );
    setupMatrix( baseFactor );

    printMatrix( "Matriz", baseMatrix, width, height ); 
    printMatrix( "Fatores", baseFactor, width, 1 ); 

    size_t index = 0;
    while( exec[index].transform_ )
    {
        t_dataVector matrix( baseMatrix );
        t_dataVector factor( baseFactor );

        // Warmup (arruma o cache, carrega OpenMP, etc).
        exec[index].transform_( matrix, factor );

        timer.start();
        for(int i = 0; i < 50; ++i )
        {
            exec[index].transform_( matrix, factor );
        }
        timer.stop();
        printMatrix( "Matriz", matrix, width, height ); 
        printMatrix( "Fatores", factor, width, 1 ); 
        std::cout << exec[index].name_ << " time: " << timer.format() << std::endl;

        ++index;
    }
    return 0;
}
