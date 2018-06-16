/*
 * MicroHH
 * Copyright (c) 2011-2017 Chiel van Heerwaarden
 * Copyright (c) 2011-2017 Thijs Heus
 * Copyright (c) 2014-2017 Bart van Stratum
 *
 * This file is part of MicroHH
 *
 * MicroHH is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * MicroHH is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with MicroHH.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef FINITE_DIFFERENCE

// In case the code is compiled with NVCC, add the macros for CUDA
#ifdef __CUDACC__
#  define CUDA_MACRO __host__ __device__
#else
#  define CUDA_MACRO
#endif

namespace Finite_difference
{
    namespace O2
    {
        template<typename TF>
        CUDA_MACRO inline TF interp2(const TF a, const TF b)
        {
            return TF(0.5) * (a + b);
        }

        template<typename TF>
        CUDA_MACRO inline TF interp22(const TF a, const TF b, const TF c, const TF d)
        {
            return TF(0.25) * (a + b + c + d);
        }

        template<typename TF>
        CUDA_MACRO inline TF grad2x(const TF a, const TF b)
        {
            return (b - a);
        }
    }

    namespace O4
    {
        template<typename TF> constexpr TF ci0  = TF(-1./16.);
        template<typename TF> constexpr TF ci1  = TF( 9./16.);
        template<typename TF> constexpr TF ci2  = TF( 9./16.);
        template<typename TF> constexpr TF ci3  = TF(-1./16.);

        template<typename TF> constexpr TF bi0  = TF( 5./16.);
        template<typename TF> constexpr TF bi1  = TF(15./16.);
        template<typename TF> constexpr TF bi2  = TF(-5./16.);
        template<typename TF> constexpr TF bi3  = TF( 1./16.);

        template<typename TF> constexpr TF ti0  = TF( 1./16.);
        template<typename TF> constexpr TF ti1  = TF(-5./16.);
        template<typename TF> constexpr TF ti2  = TF(15./16.);
        template<typename TF> constexpr TF ti3  = TF( 5./16.);

        template<typename TF> constexpr TF cg0  = TF(  1.);
        template<typename TF> constexpr TF cg1  = TF(-27.);
        template<typename TF> constexpr TF cg2  = TF( 27.);
        template<typename TF> constexpr TF cg3  = TF( -1.);
        template<typename TF> constexpr TF cgi  = TF(  1./24.);

        template<typename TF> constexpr TF bg0  = TF(-23.);
        template<typename TF> constexpr TF bg1  = TF( 21.);
        template<typename TF> constexpr TF bg2  = TF(  3.);
        template<typename TF> constexpr TF bg3  = TF( -1.);

        template<typename TF> constexpr TF tg0  = TF(  1.);
        template<typename TF> constexpr TF tg1  = TF( -3.);
        template<typename TF> constexpr TF tg2  = TF(-21.);
        template<typename TF> constexpr TF tg3  = TF( 23.);

        template<typename TF> constexpr TF cdg0 = TF(-1460./576.);
        template<typename TF> constexpr TF cdg1 = TF(  783./576.);
        template<typename TF> constexpr TF cdg2 = TF(  -54./576.);
        template<typename TF> constexpr TF cdg3 = TF(    1./576.);

        template<typename TF>
        CUDA_MACRO inline TF interp4(const TF a, const TF b, const TF c, const TF d)
        {
            return ci0<TF>*a + ci1<TF>*b + ci2<TF>*c + ci3<TF>*d;
        }

        template<typename TF>
        CUDA_MACRO inline TF interp4bot(const TF a, const TF b, const TF c, const TF d)
        {
            return bi0<TF>*a + bi1<TF>*b - bi2<TF>*c + bi3<TF>*d;
        }

        template<typename TF>
        CUDA_MACRO inline TF interp4top(const TF a, const TF b, const TF c, const TF d)
        {
            return ti0<TF>*a + ti1<TF>*b + ti2<TF>*c + ti3<TF>*d;
        }

        template<typename TF>
        CUDA_MACRO inline TF grad4(const TF a, const TF b, const TF c, const TF d, const TF dxi)
        {
            return ( -TF(1./24.)*(d-a) + TF(27./24.)*(c-b) ) * dxi;
        }

        template<typename TF>
        CUDA_MACRO inline TF grad4x(const TF a, const TF b, const TF c, const TF d)
        {
            return (-(d-a) + TF(27.)*(c-b));
        }
    }
}
#endif
