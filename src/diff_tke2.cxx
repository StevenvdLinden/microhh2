/*
 * MicroHH
 * Copyright (c) 2011-2018 Chiel van Heerwaarden
 * Copyright (c) 2011-2018 Thijs Heus
 * Copyright (c) 2014-2018 Bart van Stratum
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

#include <algorithm>
#include <cmath>
#include <iostream>

#include "grid.h"
#include "fields.h"
#include "master.h"
#include "defines.h"
#include "constants.h"
#include "monin_obukhov.h"
#include "thermo.h"
#include "boundary.h"

// yet to be done
#include "stats.h"
#include "advec.h"
//
#include "diff_tke2.h"

namespace
{
    namespace most = Monin_obukhov;

    // Define constants used in subgrid tke scheme
    const TF ap  = 1.5;
    const TF cf  = 2.5;
    const TF ce1 = 0.19;
    const TF ce2 = 0.51;
    const TF cm  = 0.12;
    const TF ch1 = 1.0;
    const TF ch2 = 2.0;
    const TF cn  = 0.76;

    const TF n_mason   = 2.0; // value of Mason-Thompson wall correction
    const TF sgstkemin = 1e-9; // minimum value of SGS TKE to prevent model crash

    enum class Surface_model {Enabled, Disabled};

    // NOOT: deze zou al goed moeten zijn
    template <typename TF, Surface_model surface_model>
    void calc_strain2(TF* restrict strain2,
                      TF* restrict u, TF* restrict v, TF* restrict w,
                      TF* restrict ufluxbot, TF* restrict vfluxbot,
                      TF* restrict ustar, TF* restrict obuk,
                      const TF* restrict z, const TF* restrict dzi, const TF* restrict dzhi,
                      const TF dxi, const TF dyi,
                      const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
                      const int jj, const int kk)
    {
        const int ii = 1;
        const int k_offset = (surface_model == Surface_model::Disabled) ? 0 : 1;

        // If the wall isn't resolved, calculate du/dz and dv/dz at lowest grid height using MO
        if (surface_model == Surface_model::Enabled)
        {
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij  = i + j*jj;
                    const int ijk = i + j*jj + kstart*kk;

                    strain2[ijk] = TF(2.)*(
                                   // du/dx + du/dx
                                   + std::pow((u[ijk+ii]-u[ijk])*dxi, TF(2))

                                   // dv/dy + dv/dy
                                   + std::pow((v[ijk+jj]-v[ijk])*dyi, TF(2))

                                   // dw/dz + dw/dz
                                   + std::pow((w[ijk+kk]-w[ijk])*dzi[kstart], TF(2))

                                   // du/dy + dv/dx
                                   + TF(0.125)*std::pow((u[ijk      ]-u[ijk   -jj])*dyi  + (v[ijk      ]-v[ijk-ii   ])*dxi, TF(2))
                                   + TF(0.125)*std::pow((u[ijk+ii   ]-u[ijk+ii-jj])*dyi  + (v[ijk+ii   ]-v[ijk      ])*dxi, TF(2))
                                   + TF(0.125)*std::pow((u[ijk   +jj]-u[ijk      ])*dyi  + (v[ijk   +jj]-v[ijk-ii+jj])*dxi, TF(2))
                                   + TF(0.125)*std::pow((u[ijk+ii+jj]-u[ijk+ii   ])*dyi  + (v[ijk+ii+jj]-v[ijk   +jj])*dxi, TF(2))

                                   // du/dz
                                   + TF(0.5)*std::pow(TF(-0.5)*(ufluxbot[ij]+ufluxbot[ij+ii])/(Constants::kappa<TF>*z[kstart]*ustar[ij])*most::phim(z[kstart]/obuk[ij]), TF(2.))

                                   // dw/dx
                                   + TF(0.125)*std::pow((w[ijk      ]-w[ijk-ii   ])*dxi, TF(2))
                                   + TF(0.125)*std::pow((w[ijk+ii   ]-w[ijk      ])*dxi, TF(2))
                                   + TF(0.125)*std::pow((w[ijk   +kk]-w[ijk-ii+kk])*dxi, TF(2))
                                   + TF(0.125)*std::pow((w[ijk+ii+kk]-w[ijk   +kk])*dxi, TF(2))

                                   // dv/dz
                                   + TF(0.5)*std::pow(TF(-0.5)*(vfluxbot[ij]+vfluxbot[ij+jj])/(Constants::kappa<TF>*z[kstart]*ustar[ij])*most::phim(z[kstart]/obuk[ij]), TF(2))

                                   // dw/dy
                                   + TF(0.125)*std::pow((w[ijk      ]-w[ijk-jj   ])*dyi, TF(2))
                                   + TF(0.125)*std::pow((w[ijk+jj   ]-w[ijk      ])*dyi, TF(2))
                                   + TF(0.125)*std::pow((w[ijk   +kk]-w[ijk-jj+kk])*dyi, TF(2))
                                   + TF(0.125)*std::pow((w[ijk+jj+kk]-w[ijk   +kk])*dyi, TF(2)) );

                    // add a small number to avoid zero divisions
                    strain2[ijk] += Constants::dsmall;
                }
        }

        for (int k=kstart+k_offset; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    strain2[ijk] = TF(2.)*(
                                   // du/dx + du/dx
                                   + std::pow((u[ijk+ii]-u[ijk])*dxi, TF(2))

                                   // dv/dy + dv/dy
                                   + std::pow((v[ijk+jj]-v[ijk])*dyi, TF(2))

                                   // dw/dz + dw/dz
                                   + std::pow((w[ijk+kk]-w[ijk])*dzi[k], TF(2))

                                   // du/dy + dv/dx
                                   + TF(0.125)*std::pow((u[ijk      ]-u[ijk   -jj])*dyi  + (v[ijk      ]-v[ijk-ii   ])*dxi, TF(2))
                                   + TF(0.125)*std::pow((u[ijk+ii   ]-u[ijk+ii-jj])*dyi  + (v[ijk+ii   ]-v[ijk      ])*dxi, TF(2))
                                   + TF(0.125)*std::pow((u[ijk   +jj]-u[ijk      ])*dyi  + (v[ijk   +jj]-v[ijk-ii+jj])*dxi, TF(2))
                                   + TF(0.125)*std::pow((u[ijk+ii+jj]-u[ijk+ii   ])*dyi  + (v[ijk+ii+jj]-v[ijk   +jj])*dxi, TF(2))

                                   // du/dz + dw/dx
                                   + TF(0.125)*std::pow((u[ijk      ]-u[ijk   -kk])*dzhi[k  ] + (w[ijk      ]-w[ijk-ii   ])*dxi, TF(2))
                                   + TF(0.125)*std::pow((u[ijk+ii   ]-u[ijk+ii-kk])*dzhi[k  ] + (w[ijk+ii   ]-w[ijk      ])*dxi, TF(2))
                                   + TF(0.125)*std::pow((u[ijk   +kk]-u[ijk      ])*dzhi[k+1] + (w[ijk   +kk]-w[ijk-ii+kk])*dxi, TF(2))
                                   + TF(0.125)*std::pow((u[ijk+ii+kk]-u[ijk+ii   ])*dzhi[k+1] + (w[ijk+ii+kk]-w[ijk   +kk])*dxi, TF(2))

                                   // dv/dz + dw/dy
                                   + TF(0.125)*std::pow((v[ijk      ]-v[ijk   -kk])*dzhi[k  ] + (w[ijk      ]-w[ijk-jj   ])*dyi, TF(2))
                                   + TF(0.125)*std::pow((v[ijk+jj   ]-v[ijk+jj-kk])*dzhi[k  ] + (w[ijk+jj   ]-w[ijk      ])*dyi, TF(2))
                                   + TF(0.125)*std::pow((v[ijk   +kk]-v[ijk      ])*dzhi[k+1] + (w[ijk   +kk]-w[ijk-jj+kk])*dyi, TF(2))
                                   + TF(0.125)*std::pow((v[ijk+jj+kk]-v[ijk+jj   ])*dzhi[k+1] + (w[ijk+jj+kk]-w[ijk   +kk])*dyi, TF(2)) );

                           // Add a small number to avoid zero divisions.
                           strain2[ijk] += Constants::dsmall;
                }
    }

    // NOOT: deze zou ook af moeten zijn, (1) enige verschil Surface_model is dat in het ene geval wel een Mason-wall correctie wordt toegepast, en in de andere niet. (2) omdat er geen thermo is, wordt toch overal meteen mvisc bij opgeteld
    template <typename TF, Surface_model surface_model>
    void calc_evisc_neutral(TF* restrict evisc,
                            TF* restrict sgstke, //toegevoegd
                            TF* restrict u, TF* restrict v, TF* restrict w,
                            TF* restrict ufluxbot, TF* restrict vfluxbot,
                            const TF* restrict z, const TF* restrict dz, const TF z0m, const TF mvisc,
                            const TF dx, const TF dy,
                            const TF cm, const TF cn, const TF n_mason, //toegevoegd
                            const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
                            const int icells, const int jcells, const int ijcells,
                            Boundary_cyclic<TF>& boundary_cyclic)
    {
        const int jj = icells;
        const int kk = ijcells;

        if (surface_model == Surface_model::Disabled)
        {
            for (int k=kstart; k<kend; ++k)
            {
                const TF mlen0 = std::pow(dx*dy*dz[k], TF(1.)/TF(3.));
                TF fac;

                for (int j=jstart; j<jend; ++j)
                    #pragma ivdep
                    for (int i=istart; i<iend; ++i)
                    {
                        const int ijk = i + j*jj + k*kk;
                        TF mlen = cn * std::sqrt(sgstke[ijk]);
                        fac = std::min(mlen0, mlen);

                        evisc[ijk] = cm * fac * std::sqrt(sgstke[ijk]) + mvisc;
                    }
            }

            boundary_cyclic.exec(evisc);

            // NOOT: waarom is dit uitgecomment?
            /*
            // For a resolved wall the viscosity at the wall is needed. For now, assume that the eddy viscosity
            // is zero, so set ghost cell such that the viscosity interpolated to the surface equals the molecular viscosity.
            const int kb = kstart;
            const int kt = kend-1;
            for (int j=0; j<jcells; ++j)
                #pragma ivdep
                for (int i=0; i<icells; ++i)
                {
                    const int ijkb = i + j*jj + kb*kk;
                    const int ijkt = i + j*jj + kt*kk;
                    evisc[ijkb-kk] = 2 * mvisc - evisc[ijkb];
                    evisc[ijkt+kk] = 2 * mvisc - evisc[ijkt];
                }
                */
        }
        else
        {
            for (int k=kstart; k<kend; ++k)
            {
                // Calculate smagorinsky constant times filter width squared, use wall damping according to Mason's paper.
                const TF mlen0 = std::pow(dx*dy*dz[k], TF(1./3.));
                TF fac;

                for (int j=jstart; j<jend; ++j)
                    #pragma ivdep
                    for (int i=istart; i<iend; ++i)
                    {
                        const int ijk = i + j*jj + k*kk;
                        TF mlen = cn * std::sqrt(sgstke[ijk]);
                        fac = std::min(mlen0, mlen);
                        fac = std::pow(TF(1.)/(TF(1.)/std::pow(fac, n_mason) + TF(1.)/(std::pow(Constants::kappa<TF>*(z[k]+z0m), n_mason))), TF(1.)/n_mason);

                        evisc[ijk] = cm * fac * std::sqrt(sgstke[ijk]) + mvisc;
                    }
            }

            boundary_cyclic.exec(evisc);
        }
    }

    // NOOT: deze zou af moeten zijn
    template<typename TF, Surface_model surface_model>
    void calc_evisc(TF* restrict evisc,
                    TF* restruct sgstke, //toegevoegd
                    TF* restrict u, TF* restrict v, TF* restrict w,  TF* restrict N2,
                    TF* restrict ufluxbot, TF* restrict vfluxbot, TF* restrict bfluxbot,
                    TF* restrict ustar, TF* restrict obuk,
                    const TF* restrict z, const TF* restrict dz, const TF* restrict dzi,
                    const TF dx, const TF dy,
                    const TF z0m, const TF mvisc,
                    const TF cm, const TF cn, const TF n_mason, //toegevoegd
                    const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
                    const int icells, const int jcells, const int ijcells,
                    Boundary_cyclic<TF>& boundary_cyclic)
    {
        const int jj = icells;
        const int kk = ijcells;

        TF fac;

        if (surface_model == Surface_model::Disabled)
        {
            for (int k=kstart; k<kend; ++k)
            {
                // calculate geometric mean of filter mesh size based on Deardorff, 1973
                const TF mlen0 = std::pow(dx*dy*dz[k], TF(1.)/TF(3.));
                //const TF fac = std::pow(mlen, TF(2));

                for (int j=jstart; j<jend; ++j)
                    #pragma ivdep
                    for (int i=istart; i<iend; ++i)
                    {
                        const int ijk = i + j*jj + k*kk;

                        // Calculate eddy viscosity for momentum based on Deardorff, 1980
                        TF mlen  = mlen0; // re-declare each loop-iteration
                        if( N2[ijk] > TF(0.)) mlen = cn * std::sqrt(sgstke[ijk]) / std::sqrt(std::abs(N2[ijk]));
                        fac  = std::min(mlen0, mlen);
                        // do not use wall-damping with resolved boundaries
                        //fac  = std::pow(1./(1./std::pow(fac, nm) + 1./(std::pow(Constants::kappa*(z[k]+z0m), nm))), 1./nm);

                        evisc[ijk] = cm * fac * std::sqrt(sgstke[ijk]); //+ mvisc; NOOT: wel of niet optellen?

                    }
            }

            // For a resolved wall the viscosity at the wall is needed. For now, assume that the eddy viscosity
            // is zero, so set ghost cell such that the viscosity interpolated to the surface equals the molecular viscosity.
            const int kb = kstart;
            const int kt = kend-1;
            for (int j=0; j<jcells; ++j)
                #pragma ivdep
                for (int i=0; i<icells; ++i)
                {
                    const int ijkb = i + j*jj + kb*kk;
                    const int ijkt = i + j*jj + kt*kk;
                    evisc[ijkb-kk] = evisc[ijkb];
                    evisc[ijkt+kk] = evisc[ijkt];
                }
        }
        else
        {
            for (int k=kstart; k<kend; ++k)
            {
                // calculate geometric mean of filter mesh size based on Deardorff, 1973
                const TF mlen0 = std::pow(dx*dy*dz[k], TF(1.)/TF(3.));

                for (int j=jstart; j<jend; ++j)
                    #pragma ivdep
                    for (int i=istart; i<iend; ++i)
                    {
                        const int ijk = i + j*jj + k*kk;
                        // Calculate eddy viscosity for momentum based on Deardorff, 1980
                        TF mlen  = mlen0; // re-declare each loop-iteration
                        if( N2[ijk] > TF(0.)) mlen = cn * std::sqrt(sgstke[ijk]) / std::sqrt(std::abs(N2[ijk]));
                        fac  = std::min(mlen0, mlen);
                        fac  = std::pow(TF(1.)/(TF(1.)/std::pow(fac, n_mason) + TF(1.)/(std::pow(Constants::kappa<TF>*(z[k]+z0m), n_mason))), TF(1.)/n_mason);

                        evisc[ijk] = cm * fac * std::sqrt(sgstke[ijk]); //+ mvisc; NOOT: wel of niet optellen?
                    }
            }
        }

        boundary_cyclic.exec(evisc);
    }

    // NOOT: deze zou al goed moeten zijn
    template <typename TF, Surface_model surface_model>
    void diff_u(TF* restrict ut,
                const TF* restrict u, const TF* restrict v, const TF* restrict w,
                const TF* restrict dzi, const TF* restrict dzhi, const TF dxi, const TF dyi,
                const TF* restrict evisc,
                const TF* restrict fluxbot, const TF* restrict fluxtop,
                const TF* restrict rhoref, const TF* restrict rhorefh,
                const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
                const int jj, const int kk)
    {
        const int ii = 1;

        TF eviscn, eviscs, eviscb, evisct;

        const int k_offset = (surface_model == Surface_model::Disabled) ? 0 : 1;

        if (surface_model == Surface_model::Enabled)
        {
            // bottom boundary
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij  = i + j*jj;
                    const int ijk = i + j*jj + kstart*kk;
                    eviscn = TF(0.25)*(evisc[ijk-ii   ] + evisc[ijk   ] + evisc[ijk-ii+jj] + evisc[ijk+jj]);
                    eviscs = TF(0.25)*(evisc[ijk-ii-jj] + evisc[ijk-jj] + evisc[ijk-ii   ] + evisc[ijk   ]);
                    evisct = TF(0.25)*(evisc[ijk-ii   ] + evisc[ijk   ] + evisc[ijk-ii+kk] + evisc[ijk+kk]);
                    eviscb = TF(0.25)*(evisc[ijk-ii-kk] + evisc[ijk-kk] + evisc[ijk-ii   ] + evisc[ijk   ]);

                    ut[ijk] +=
                             // du/dx + du/dx
                             + ( evisc[ijk   ]*(u[ijk+ii]-u[ijk   ])*dxi
                               - evisc[ijk-ii]*(u[ijk   ]-u[ijk-ii])*dxi ) * TF(2.)*dxi
                             // du/dy + dv/dx
                             + ( eviscn*((u[ijk+jj]-u[ijk   ])*dyi + (v[ijk+jj]-v[ijk-ii+jj])*dxi)
                               - eviscs*((u[ijk   ]-u[ijk-jj])*dyi + (v[ijk   ]-v[ijk-ii   ])*dxi) ) * dyi
                             // du/dz + dw/dx
                             + ( rhorefh[kstart+1] * evisct*((u[ijk+kk]-u[ijk   ])* dzhi[kstart+1] + (w[ijk+kk]-w[ijk-ii+kk])*dxi)
                               + rhorefh[kstart  ] * fluxbot[ij] ) / rhoref[kstart] * dzi[kstart];
                }

            // top boundary
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij  = i + j*jj;
                    const int ijk = i + j*jj + (kend-1)*kk;
                    eviscn = TF(0.25)*(evisc[ijk-ii   ] + evisc[ijk   ] + evisc[ijk-ii+jj] + evisc[ijk+jj]);
                    eviscs = TF(0.25)*(evisc[ijk-ii-jj] + evisc[ijk-jj] + evisc[ijk-ii   ] + evisc[ijk   ]);
                    evisct = TF(0.25)*(evisc[ijk-ii   ] + evisc[ijk   ] + evisc[ijk-ii+kk] + evisc[ijk+kk]);
                    eviscb = TF(0.25)*(evisc[ijk-ii-kk] + evisc[ijk-kk] + evisc[ijk-ii   ] + evisc[ijk   ]);
                    ut[ijk] +=
                             // du/dx + du/dx
                             + ( evisc[ijk   ]*(u[ijk+ii]-u[ijk   ])*dxi
                               - evisc[ijk-ii]*(u[ijk   ]-u[ijk-ii])*dxi ) * TF(2.)*dxi
                             // du/dy + dv/dx
                             + ( eviscn*((u[ijk+jj]-u[ijk   ])*dyi  + (v[ijk+jj]-v[ijk-ii+jj])*dxi)
                               - eviscs*((u[ijk   ]-u[ijk-jj])*dyi  + (v[ijk   ]-v[ijk-ii   ])*dxi) ) * dyi
                             // du/dz + dw/dx
                             + (- rhorefh[kend  ] * fluxtop[ij]
                                - rhorefh[kend-1] * eviscb*((u[ijk   ]-u[ijk-kk])* dzhi[kend-1] + (w[ijk   ]-w[ijk-ii   ])*dxi) ) / rhoref[kend-1] * dzi[kend-1];
                }
        }

        for (int k=kstart+k_offset; k<kend-k_offset; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    eviscn = TF(0.25)*(evisc[ijk-ii   ] + evisc[ijk   ] + evisc[ijk-ii+jj] + evisc[ijk+jj]);
                    eviscs = TF(0.25)*(evisc[ijk-ii-jj] + evisc[ijk-jj] + evisc[ijk-ii   ] + evisc[ijk   ]);
                    evisct = TF(0.25)*(evisc[ijk-ii   ] + evisc[ijk   ] + evisc[ijk-ii+kk] + evisc[ijk+kk]);
                    eviscb = TF(0.25)*(evisc[ijk-ii-kk] + evisc[ijk-kk] + evisc[ijk-ii   ] + evisc[ijk   ]);
                    ut[ijk] +=
                             // du/dx + du/dx
                             + ( evisc[ijk   ]*(u[ijk+ii]-u[ijk   ])*dxi
                               - evisc[ijk-ii]*(u[ijk   ]-u[ijk-ii])*dxi ) * TF(2.)*dxi
                             // du/dy + dv/dx
                             + ( eviscn*((u[ijk+jj]-u[ijk   ])*dyi  + (v[ijk+jj]-v[ijk-ii+jj])*dxi)
                               - eviscs*((u[ijk   ]-u[ijk-jj])*dyi  + (v[ijk   ]-v[ijk-ii   ])*dxi) ) * dyi
                             // du/dz + dw/dx
                             + ( rhorefh[k+1] * evisct*((u[ijk+kk]-u[ijk   ])* dzhi[k+1] + (w[ijk+kk]-w[ijk-ii+kk])*dxi)
                               - rhorefh[k  ] * eviscb*((u[ijk   ]-u[ijk-kk])* dzhi[k  ] + (w[ijk   ]-w[ijk-ii   ])*dxi) ) / rhoref[k] * dzi[k];
                }
    }

    // NOOT: deze zou al goed moeten zijn
    template <typename TF, Surface_model surface_model>
    void diff_v(TF* restrict vt,
                const TF* restrict u, const TF* restrict v, const TF* restrict w,
                const TF* restrict dzi, const TF* restrict dzhi, const TF dxi, const TF dyi,
                const TF* restrict evisc,
                TF* restrict fluxbot, TF* restrict fluxtop,
                TF* restrict rhoref, TF* restrict rhorefh,
                const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
                const int jj, const int kk)

    {
        const int ii = 1;

        TF evisce, eviscw, eviscb, evisct;

        const int k_offset = (surface_model == Surface_model::Disabled) ? 0 : 1;

        if (surface_model == Surface_model::Enabled)
        {
            // bottom boundary
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij  = i + j*jj;
                    const int ijk = i + j*jj + kstart*kk;
                    evisce = TF(0.25)*(evisc[ijk   -jj] + evisc[ijk   ] + evisc[ijk+ii-jj] + evisc[ijk+ii]);
                    eviscw = TF(0.25)*(evisc[ijk-ii-jj] + evisc[ijk-ii] + evisc[ijk   -jj] + evisc[ijk   ]);
                    evisct = TF(0.25)*(evisc[ijk   -jj] + evisc[ijk   ] + evisc[ijk+kk-jj] + evisc[ijk+kk]);
                    eviscb = TF(0.25)*(evisc[ijk-kk-jj] + evisc[ijk-kk] + evisc[ijk   -jj] + evisc[ijk   ]);
                    vt[ijk] +=
                             // dv/dx + du/dy
                             + ( evisce*((v[ijk+ii]-v[ijk   ])*dxi + (u[ijk+ii]-u[ijk+ii-jj])*dyi)
                               - eviscw*((v[ijk   ]-v[ijk-ii])*dxi + (u[ijk   ]-u[ijk   -jj])*dyi) ) * dxi
                             // dv/dy + dv/dy
                             + ( evisc[ijk   ]*(v[ijk+jj]-v[ijk   ])*dyi
                               - evisc[ijk-jj]*(v[ijk   ]-v[ijk-jj])*dyi ) * TF(2.)*dyi
                             // dv/dz + dw/dy
                             + ( rhorefh[kstart+1] * evisct*((v[ijk+kk]-v[ijk   ])*dzhi[kstart+1] + (w[ijk+kk]-w[ijk-jj+kk])*dyi)
                               + rhorefh[kstart  ] * fluxbot[ij] ) / rhoref[kstart] * dzi[kstart];
                }

            // top boundary
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij  = i + j*jj;
                    const int ijk = i + j*jj + (kend-1)*kk;
                    evisce = TF(0.25)*(evisc[ijk   -jj] + evisc[ijk   ] + evisc[ijk+ii-jj] + evisc[ijk+ii]);
                    eviscw = TF(0.25)*(evisc[ijk-ii-jj] + evisc[ijk-ii] + evisc[ijk   -jj] + evisc[ijk   ]);
                    evisct = TF(0.25)*(evisc[ijk   -jj] + evisc[ijk   ] + evisc[ijk+kk-jj] + evisc[ijk+kk]);
                    eviscb = TF(0.25)*(evisc[ijk-kk-jj] + evisc[ijk-kk] + evisc[ijk   -jj] + evisc[ijk   ]);
                    vt[ijk] +=
                             // dv/dx + du/dy
                             + ( evisce*((v[ijk+ii]-v[ijk   ])*dxi + (u[ijk+ii]-u[ijk+ii-jj])*dyi)
                               - eviscw*((v[ijk   ]-v[ijk-ii])*dxi + (u[ijk   ]-u[ijk   -jj])*dyi) ) * dxi
                             // dv/dy + dv/dy
                             + ( evisc[ijk   ]*(v[ijk+jj]-v[ijk   ])*dyi
                               - evisc[ijk-jj]*(v[ijk   ]-v[ijk-jj])*dyi ) * TF(2.)*dyi
                             // dv/dz + dw/dy
                             + (- rhorefh[kend  ] * fluxtop[ij]
                                - rhorefh[kend-1] * eviscb*((v[ijk   ]-v[ijk-kk])*dzhi[kend-1] + (w[ijk   ]-w[ijk-jj   ])*dyi) ) / rhoref[kend-1] * dzi[kend-1];
                }
        }

        for (int k=kstart+k_offset; k<kend-k_offset; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    evisce = TF(0.25)*(evisc[ijk   -jj] + evisc[ijk   ] + evisc[ijk+ii-jj] + evisc[ijk+ii]);
                    eviscw = TF(0.25)*(evisc[ijk-ii-jj] + evisc[ijk-ii] + evisc[ijk   -jj] + evisc[ijk   ]);
                    evisct = TF(0.25)*(evisc[ijk   -jj] + evisc[ijk   ] + evisc[ijk+kk-jj] + evisc[ijk+kk]);
                    eviscb = TF(0.25)*(evisc[ijk-kk-jj] + evisc[ijk-kk] + evisc[ijk   -jj] + evisc[ijk   ]);
                    vt[ijk] +=
                             // dv/dx + du/dy
                             + ( evisce*((v[ijk+ii]-v[ijk   ])*dxi + (u[ijk+ii]-u[ijk+ii-jj])*dyi)
                               - eviscw*((v[ijk   ]-v[ijk-ii])*dxi + (u[ijk   ]-u[ijk   -jj])*dyi) ) * dxi
                             // dv/dy + dv/dy
                             + ( evisc[ijk   ]*(v[ijk+jj]-v[ijk   ])*dyi
                               - evisc[ijk-jj]*(v[ijk   ]-v[ijk-jj])*dyi ) * TF(2.)*dyi
                             // dv/dz + dw/dy
                             + ( rhorefh[k+1] * evisct*((v[ijk+kk]-v[ijk   ])*dzhi[k+1] + (w[ijk+kk]-w[ijk-jj+kk])*dyi)
                               - rhorefh[k  ] * eviscb*((v[ijk   ]-v[ijk-kk])*dzhi[k  ] + (w[ijk   ]-w[ijk-jj   ])*dyi) ) / rhoref[k] * dzi[k];
                }
    }

    // NOOT: deze zou al goed moeten zijn
    template <typename TF>
    void diff_w(TF* restrict wt,
                const TF* restrict u, const TF* restrict v, const TF* restrict w,
                const TF* restrict dzi, const TF* restrict dzhi, const TF dxi, const TF dyi,
                const TF* restrict evisc,
                const TF* restrict rhoref, const TF* restrict rhorefh,
                const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
                const int jj, const int kk)
    {
        const int ii = 1;

        TF evisce, eviscw, eviscn, eviscs;

        for (int k=kstart+1; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    evisce = TF(0.25)*(evisc[ijk   -kk] + evisc[ijk   ] + evisc[ijk+ii-kk] + evisc[ijk+ii]);
                    eviscw = TF(0.25)*(evisc[ijk-ii-kk] + evisc[ijk-ii] + evisc[ijk   -kk] + evisc[ijk   ]);
                    eviscn = TF(0.25)*(evisc[ijk   -kk] + evisc[ijk   ] + evisc[ijk+jj-kk] + evisc[ijk+jj]);
                    eviscs = TF(0.25)*(evisc[ijk-jj-kk] + evisc[ijk-jj] + evisc[ijk   -kk] + evisc[ijk   ]);
                    wt[ijk] +=
                             // dw/dx + du/dz
                             + ( evisce*((w[ijk+ii]-w[ijk   ])*dxi + (u[ijk+ii]-u[ijk+ii-kk])*dzhi[k])
                               - eviscw*((w[ijk   ]-w[ijk-ii])*dxi + (u[ijk   ]-u[ijk+  -kk])*dzhi[k]) ) * dxi
                             // dw/dy + dv/dz
                             + ( eviscn*((w[ijk+jj]-w[ijk   ])*dyi + (v[ijk+jj]-v[ijk+jj-kk])*dzhi[k])
                               - eviscs*((w[ijk   ]-w[ijk-jj])*dyi + (v[ijk   ]-v[ijk+  -kk])*dzhi[k]) ) * dyi
                             // dw/dz + dw/dz
                             + ( rhoref[k  ] * evisc[ijk   ]*(w[ijk+kk]-w[ijk   ])*dzi[k  ]
                               - rhoref[k-1] * evisc[ijk-kk]*(w[ijk   ]-w[ijk-kk])*dzi[k-1] ) / rhorefh[k] * TF(2.)*dzhi[k];
                }
    }

    // deze zou af moeten zijn
    template <typename TF, Surface_model surface_model>
    void diff_c(TF* restrict at, const TF* restrict a,
                const TF* restrict dzi, const TF* restrict dzhi, const TF dxidxi, const TF dyidyi,
                const TF* restrict evisc, const TF* restrict sgstke, const TF* restrict N2, //sgstke en N2 toegevoegd
                const TF* restrict fluxbot, const TF* restrict fluxtop,
                const TF* restrict rhoref, const TF* restrict rhorefh,
                const TF cn, const TF ch1, const TF ch2, const TF n_mason, //toegevoegd
                const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
                const int jj, const int kk)
    {
        const int ii = 1;

        TF evisce, eviscw, eviscn, eviscs, evisct, eviscb;
        TF fac, tPri;

        const int k_offset = (surface_model == Surface_model::Disabled) ? 0 : 1;

        if (surface_model == Surface_model::Enabled)
        {
            // bottom boundary
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij  = i + j*jj;
                    const int ijk = i + j*jj + kstart*kk;

                    // calculate geometric mean of filter mesh size based on Deardorff, 1973
                    const TF mlen0 = std::pow(dx*dy*dz[kstart], TF(1.)/TF(3.));

                    // Calculate turbulent length scale based on Deardorff, 1980
                    TF mlen  = mlen0;
                    if ( N2[ijk] > TF(0.)) mlen = cn * std::sqrt(sgstke[ijk]) / std::sqrt(std::abs(N2[ijk]));
                    fac  = std::min(mlen0, mlen);
                    fac  = std::pow(TF(1.)/(TF(1.)/std::pow(fac, n_mason) + TF(1.)/(std::pow(Constants::kappa<TF>*(z[k]+z0m), n_mason))), TF(1.)/n_mason);

                    // Calculate the inverse stability dependent turbulent Prandtl number
                    tPri = (ch1 + ch2 * fac / mlen0);

                    evisce = TF(0.5)*(evisc[ijk   ]+evisc[ijk+ii]) * tPri;
                    eviscw = TF(0.5)*(evisc[ijk-ii]+evisc[ijk   ]) * tPri;
                    eviscn = TF(0.5)*(evisc[ijk   ]+evisc[ijk+jj]) * tPri;
                    eviscs = TF(0.5)*(evisc[ijk-jj]+evisc[ijk   ]) * tPri;
                    evisct = TF(0.5)*(evisc[ijk   ]+evisc[ijk+kk]) * tPri;
                    eviscb = TF(0.5)*(evisc[ijk-kk]+evisc[ijk   ]) * tPri;

                    at[ijk] +=
                             + ( evisce*(a[ijk+ii]-a[ijk   ])
                               - eviscw*(a[ijk   ]-a[ijk-ii]) ) * dxidxi
                             + ( eviscn*(a[ijk+jj]-a[ijk   ])
                               - eviscs*(a[ijk   ]-a[ijk-jj]) ) * dyidyi
                             + ( rhorefh[kstart+1] * evisct*(a[ijk+kk]-a[ijk   ])*dzhi[kstart+1]
                               + rhorefh[kstart  ] * fluxbot[ij] ) / rhoref[kstart] * dzi[kstart];
                }

            // top boundary
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij  = i + j*jj;
                    const int ijk = i + j*jj + (kend-1)*kk;

                    // calculate geometric mean of filter mesh size based on Deardorff, 1973
                    const TF mlen0 = std::pow(dx*dy*dz[kend-1], TF(1.)/TF(3.));

                    // Calculate turbulent length scale based on Deardorff, 1980
                    TF mlen  = mlen0;
                    if( N2[ijk] > TF(0.)) mlen = cn * std::sqrt(sgstke[ijk]) / std::sqrt(std::abs(N2[ijk]));
                    fac  = std::min(mlen0, mlen);
                    fac  = std::pow(TF(1.)/(TF(1.)/std::pow(fac, n_mason) + TF(1.)/(std::pow(Constants::kappa<TF>*(z[k]+z0m), n_mason))), TF(1.)/n_mason);

                    // Calculate the inverse stability dependent turbulent Prandtl number
                    tPri = (ch1 + ch2 * fac / mlen0);

                    evisce = TF(0.5)*(evisc[ijk   ]+evisc[ijk+ii]) * tPri;
                    eviscw = TF(0.5)*(evisc[ijk-ii]+evisc[ijk   ]) * tPri;
                    eviscn = TF(0.5)*(evisc[ijk   ]+evisc[ijk+jj]) * tPri;
                    eviscs = TF(0.5)*(evisc[ijk-jj]+evisc[ijk   ]) * tPri;
                    evisct = TF(0.5)*(evisc[ijk   ]+evisc[ijk+kk]) * tPri;
                    eviscb = TF(0.5)*(evisc[ijk-kk]+evisc[ijk   ]) * tPri;

                    at[ijk] +=
                             + ( evisce*(a[ijk+ii]-a[ijk   ])
                               - eviscw*(a[ijk   ]-a[ijk-ii]) ) * dxidxi
                             + ( eviscn*(a[ijk+jj]-a[ijk   ])
                               - eviscs*(a[ijk   ]-a[ijk-jj]) ) * dyidyi
                             + (-rhorefh[kend  ] * fluxtop[ij]
                               - rhorefh[kend-1] * eviscb*(a[ijk   ]-a[ijk-kk])*dzhi[kend-1] ) / rhoref[kend-1] * dzi[kend-1];
                }
        }

        for (int k=kstart+k_offset; k<kend-k_offset; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;

                    // calculate geometric mean of filter mesh size based on Deardorff, 1973
                    const TF mlen0 = std::pow(dx*dy*dz[k], TF(1.)/TF(3.));

                    // Calculate turbulent length scale based on Deardorff, 1980
                    TF mlen  = mlen0;
                    if ( N2[ijk] > TF(0.) ) mlen = cn * std::sqrt(sgstke[ijk]) / std::sqrt(std::abs(N2[ijk]));
                    fac  = std::min(mlen0, mlen);
                    if ( k_offset == 1 ) fac  = std::pow(TF(1.)/(TF(1.)/std::pow(fac, n_mason) + TF(1.)/(std::pow(Constants::kappa<TF>*(z[k]+z0m), n_mason))), TF(1.)/n_mason); // only do wall-correction if surface model is used

                    // Calculate the inverse stability dependent turbulent Prandtl number
                    tPri = (ch1 + ch2 * fac / mlen0);

                    evisce = TF(0.5)*(evisc[ijk   ]+evisc[ijk+ii]) * tPri;
                    eviscw = TF(0.5)*(evisc[ijk-ii]+evisc[ijk   ]) * tPri;
                    eviscn = TF(0.5)*(evisc[ijk   ]+evisc[ijk+jj]) * tPri;
                    eviscs = TF(0.5)*(evisc[ijk-jj]+evisc[ijk   ]) * tPri;
                    evisct = TF(0.5)*(evisc[ijk   ]+evisc[ijk+kk]) * tPri;
                    eviscb = TF(0.5)*(evisc[ijk-kk]+evisc[ijk   ]) * tPri;

                    at[ijk] +=
                             + ( evisce*(a[ijk+ii]-a[ijk   ])
                               - eviscw*(a[ijk   ]-a[ijk-ii]) ) * dxidxi
                             + ( eviscn*(a[ijk+jj]-a[ijk   ])
                               - eviscs*(a[ijk   ]-a[ijk-jj]) ) * dyidyi
                             + ( rhorefh[k+1] * evisct*(a[ijk+kk]-a[ijk   ])*dzhi[k+1]
                               - rhorefh[k  ] * eviscb*(a[ijk   ]-a[ijk-kk])*dzhi[k]  ) / rhoref[k] * dzi[k];
                }
    }

    // deze zou af moeten zijn
    template <typename TF, Surface_model surface_model>
    void diff_s(TF* restrict sgstket, const TF* restrict sgstke,
                const TF* restrict dzi, const TF* restrict dzhi, const TF dxidxi, const TF dyidyi,
                const TF* restrict evisc,
                const TF* restrict fluxbot, const TF* restrict fluxtop,
                const TF* restrict rhoref, const TF* restrict rhorefh,
                const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
                const int jj, const int kk)
    {
        const int ii = 1;

        TF evisce, eviscw, eviscn, eviscs, evisct, eviscb;

        const int k_offset = (surface_model == Surface_model::Disabled) ? 0 : 1;

        if (surface_model == Surface_model::Enabled)
        {
            // bottom boundary
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij  = i + j*jj;
                    const int ijk = i + j*jj + kstart*kk;

                    evisce = TF(0.5)*(evisc[ijk   ]+evisc[ijk+ii]);
                    eviscw = TF(0.5)*(evisc[ijk-ii]+evisc[ijk   ]);
                    eviscn = TF(0.5)*(evisc[ijk   ]+evisc[ijk+jj]);
                    eviscs = TF(0.5)*(evisc[ijk-jj]+evisc[ijk   ]);
                    evisct = TF(0.5)*(evisc[ijk   ]+evisc[ijk+kk]);
                    eviscb = TF(0.5)*(evisc[ijk-kk]+evisc[ijk   ]);

                    sgstket[ijk] +=
                             + ( evisce*(sgstke[ijk+ii]-sgstke[ijk   ])
                               - eviscw*(sgstke[ijk   ]-sgstke[ijk-ii]) ) * dxidxi
                             + ( eviscn*(sgstke[ijk+jj]-sgstke[ijk   ])
                               - eviscs*(sgstke[ijk   ]-sgstke[ijk-jj]) ) * dyidyi
                             + ( rhorefh[kstart+1] * evisct*(sgstke[ijk+kk]-sgstke[ijk   ])*dzhi[kstart+1]
                               + rhorefh[kstart  ] * fluxbot[ij] ) / rhoref[kstart] * dzi[kstart];
                }

            // top boundary
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ij  = i + j*jj;
                    const int ijk = i + j*jj + (kend-1)*kk;

                    evisce = TF(0.5)*(evisc[ijk   ]+evisc[ijk+ii]);
                    eviscw = TF(0.5)*(evisc[ijk-ii]+evisc[ijk   ]);
                    eviscn = TF(0.5)*(evisc[ijk   ]+evisc[ijk+jj]);
                    eviscs = TF(0.5)*(evisc[ijk-jj]+evisc[ijk   ]);
                    evisct = TF(0.5)*(evisc[ijk   ]+evisc[ijk+kk]);
                    eviscb = TF(0.5)*(evisc[ijk-kk]+evisc[ijk   ]);

                    sgstket[ijk] +=
                             + ( evisce*(sgstke[ijk+ii]-sgstke[ijk   ])
                               - eviscw*(sgstke[ijk   ]-sgstke[ijk-ii]) ) * dxidxi
                             + ( eviscn*(sgstke[ijk+jj]-sgstke[ijk   ])
                               - eviscs*(sgstke[ijk   ]-sgstke[ijk-jj]) ) * dyidyi
                             + (-rhorefh[kend  ] * fluxtop[ij]
                               - rhorefh[kend-1] * eviscb*(sgstke[ijk   ]-sgstke[ijk-kk])*dzhi[kend-1] ) / rhoref[kend-1] * dzi[kend-1];
                }
        }

        for (int k=kstart+k_offset; k<kend-k_offset; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;

                    evisce = TF(0.5)*(evisc[ijk   ]+evisc[ijk+ii]);
                    eviscw = TF(0.5)*(evisc[ijk-ii]+evisc[ijk   ]);
                    eviscn = TF(0.5)*(evisc[ijk   ]+evisc[ijk+jj]);
                    eviscs = TF(0.5)*(evisc[ijk-jj]+evisc[ijk   ]);
                    evisct = TF(0.5)*(evisc[ijk   ]+evisc[ijk+kk]);
                    eviscb = TF(0.5)*(evisc[ijk-kk]+evisc[ijk   ]);

                    sgstket[ijk] +=
                             + ( evisce*(sgstke[ijk+ii]-sgstke[ijk   ])
                               - eviscw*(sgstke[ijk   ]-sgstke[ijk-ii]) ) * dxidxi
                             + ( eviscn*(sgstke[ijk+jj]-sgstke[ijk   ])
                               - eviscs*(sgstke[ijk   ]-sgstke[ijk-jj]) ) * dyidyi
                             + ( rhorefh[k+1] * evisct*(sgstke[ijk+kk]-sgstke[ijk   ])*dzhi[k+1]
                               - rhorefh[k  ] * eviscb*(sgstke[ijk   ]-sgstke[ijk-kk])*dzhi[k]  ) / rhoref[k] * dzi[k];
                }
    }

    // deze zou af moeten zijn
    template <typename TF>
    void calc_sgs_tke_shear_tend_2(TF* restrict sgstket,
                                   const TF* restrict evisc, const TF* restrict strain2,
                                   const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
                                   const int icells, const int jcells, const int ijcells,
                                   const int jj, const int kk))
    {
        const int ii = 1;

        for (int k=kstart; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;

                    // Calculate shear production of TKE based on Deardorff, 1980
                    sgstket[ijk] =  evisc[ijk] * strain2[ijk];
                }
    }

    // deze zou af moeten zijn
    template <typename TF, Surface_model surface_model>
    void calc_sgs_tke_buoyancy_tend_2(TF* restrict sgstket,  const TF* restrict sgstke,
                                      const TF* restrict evisc, const TF* restrict N2,
                                      const TF* restrict z, const TF* restrict dz,
                                      const TF z0m,
                                      const TF cn, const TF ch1, const TF ch2, const TF n_mason, //toegevoegd
                                      const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
                                      const int icells, const int jcells, const int ijcells,
                                      const int jj, const int kk))
    {
        const int ii = 1;

        TF fac, tPri;

        for (int k=kstart; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;

                    // calculate geometric mean of filter mesh size based on Deardorff, 1973
                    const TF mlen0 = std::pow(dx*dy*dz[k], TF(1.)/TF(3.));

                    // Calculate turbulent length scale based on Deardorff, 1980
                    TF mlen  = mlen0;
                    if ( N2[ijk] > TF(0.) ) mlen = cn * std::sqrt(sgstke[ijk]) / std::sqrt(std::abs(N2[ijk]));
                    fac  = std::min(mlen0, mlen);
                    if ( surface_model == Surface_model::Enabled ) fac  = std::pow(TF(1.)/(TF(1.)/std::pow(fac, n_mason) + TF(1.)/(std::pow(Constants::kappa<TF>*(z[k]+z0m), n_mason))), TF(1.)/n_mason); // only do wall-correction if surface model is used

                    // Calculate the inverse stability dependent turbulent Prandtl number
                    tPri = (ch1 + ch2 * fac / mlen0);

                    // Calculate buoyancy production of TKE based on Deardorff, 1980
                    sgstket[ijk] += -TF(1.) * evisc[ijk] * N2[ijk] * tPri;
                }
    }

    // deze zou af moeten zijn
    template <typename TF, Surface_model surface_model>
    void calc_sgs_tke_dissipation_2(TF* restrict sgstket, const TF* restrict sgstke,
                                    const TF* restrict evisc, const TF* restrict N2,
                                    const TF* restrict z, const TF* restrict dz,
                                    const TF z0m,
                                    const TF cn, const TF ce1, const TF ce2, const TF n_mason, //toegevoegd
                                    const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
                                    const int icells, const int jcells, const int ijcells,
                                    const int jj, const int kk))
    {
        const int ii = 1;

        TF fac, ce;

        for (int k=kstart; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;

                    // calculate geometric mean of filter mesh size based on Deardorff, 1973
                    const TF mlen0 = std::pow(dx*dy*dz[k], TF(1.)/TF(3.));

                    // Calculate turbulent length scale based on Deardorff, 1980
                    TF mlen  = mlen0;
                    if ( N2[ijk] > TF(0.) ) mlen = cn * std::sqrt(sgstke[ijk]) / std::sqrt(std::abs(N2[ijk]));
                    fac  = std::min(mlen0, mlen);
                    if ( surface_model == Surface_model::Enabled ) fac  = std::pow(TF(1.)/(TF(1.)/std::pow(fac, n_mason) + TF(1.)/(std::pow(Constants::kappa<TF>*(z[k]+z0m), n_mason))), TF(1.)/n_mason); // only do wall-correction if surface model is used

                    // Calculate the inverse stability dependent turbulent Prandtl number
                    ce = (ce1 + ce2 * fac / mlen0);

                    // Calculate buoyancy production of TKE based on Deardorff, 1980
                    sgstket[ijk] += -TF(1.) * ce * std::pow(sgstke[ijk], TF(3.)/TF(2.)) / fac;
                }
    }

    template<typename TF>
    TF calc_dnmul(TF* restrict evisc, const TF* restrict dzi, const TF dxidxi, const TF dyidyi, const TF tPr,
                  const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
                  const int jj, const int kk)
    {
        const TF one = 1.;
        const TF tPrfac = std::min(one, tPr);
        TF dnmul = 0;

        // get the maximum time step for diffusion
        for (int k=kstart; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    dnmul = std::max(dnmul, std::abs(tPrfac*evisc[ijk]*(dxidxi + dyidyi + dzi[k]*dzi[k])));
                }

        // get_max(&dnmul);

        return dnmul;
    }

    // deze zou af moeten zijn
    template <typename TF, Surface_model surface_model>
    void calc_prandtl(TF* restrict prandtl, const TF* restrict sgstke,
                                    const TF* restrict N2,
                                    const TF* restrict z, const TF* restrict dz,
                                    const TF z0m,
                                    const TF cn, const TF ch1, const TF ch2, const TF n_mason, //toegevoegd
                                    const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
                                    const int icells, const int jcells, const int ijcells,
                                    const int jj, const int kk))
    {
        const int ii = 1;

        TF fac, ce;

        for (int k=kstart; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;

                    // calculate geometric mean of filter mesh size based on Deardorff, 1973
                    const TF mlen0 = std::pow(dx*dy*dz[k], TF(1.)/TF(3.));

                    // Calculate turbulent length scale based on Deardorff, 1980
                    TF mlen  = mlen0;
                    if ( N2[ijk] > TF(0.) ) mlen = cn * std::sqrt(sgstke[ijk]) / std::sqrt(std::abs(N2[ijk]));
                    fac  = std::min(mlen0, mlen);
                    if ( surface_model == Surface_model::Enabled ) fac  = std::pow(TF(1.)/(TF(1.)/std::pow(fac, n_mason) + TF(1.)/(std::pow(Constants::kappa<TF>*(z[k]+z0m), n_mason))), TF(1.)/n_mason); // only do wall-correction if surface model is used

                    // Calculate the inverse stability dependent turbulent Prandtl number
                    prandtl[ijk] = TF(1.) / (ch1 + ch2 * fac / mlen0);
                }
    }

    // deze zou af moeten zijn
    template <typename TF>
    void calc_set_minimum_sgs_tke_(TF* restrict sgstke,
                                    const TF sgstkemin,
                                    const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
                                    const int icells, const int jcells, const int ijcells,
                                    const int jj, const int kk))
    {
        const int ii = 1;

        for (int k=kstart; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    sgstke[ijk] = std::min(sgstke[ijk], sgstkemin);
                }
    }
    // FIX DEZE NOG!
    // template <typename TF, Surface_model surface_model>
    // void diff_c_neutral(TF* restrict at, const TF* restrict a,
    //             const TF* restrict dzi, const TF* restrict dzhi, const TF dxidxi, const TF dyidyi,
    //             const TF* restrict evisc, const TF* restrict sgstke, const TF* restrict N2,
    //             const TF* restrict fluxbot, const TF* restrict fluxtop,
    //             const TF* restrict rhoref, const TF* restrict rhorefh, const TF tPr,
    //             const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
    //             const int jj, const int kk)
    // {
    //     const int ii = 1;
    //
    //     TF evisce, eviscw, eviscn, eviscs, evisct, eviscb;
    //     TF fac, tPri;
    //
    //     const int k_offset = (surface_model == Surface_model::Disabled) ? 0 : 1;
    //
    //     if (surface_model == Surface_model::Enabled)
    //     {
    //         // bottom boundary
    //         for (int j=jstart; j<jend; ++j)
    //             #pragma ivdep
    //             for (int i=istart; i<iend; ++i)
    //             {
    //                 const int ij  = i + j*jj;
    //                 const int ijk = i + j*jj + kstart*kk;
    //
    //                 // calculate geometric mean of filter mesh size based on Deardorff, 1973
    //                 const TF mlen0 = std::pow(dx*dy*dz[kstart], 1./3.);
    //
    //                 // Calculate turbulent length scale based on Deardorff, 1980
    //                 TF mlen  = mlen0;
    //                 if( N2[ijk] > TF(0.)) mlen = cn * std::sqrt(sgstke[ijk]) / std::sqrt(std::abs(N2[ijk]));
    //                 fac  = std::min(mlen0, mlen);
    //                 fac  = std::pow(TF(1.)/(TF(1.)/std::pow(fac, n_mason) + TF(1.)/(std::pow(Constants::kappa<TF>*(z[k]+z0m), n_mason))), TF(1.)/n_mason);
    //                 // Calculate the inverse stability dependent turbulent Prandtl number
    //                 tPri = (ch1 + ch2 * fac / mlen0);
    //
    //                 evisce = TF(0.5)*(evisc[ijk   ]+evisc[ijk+ii])/tPr;
    //                 eviscw = TF(0.5)*(evisc[ijk-ii]+evisc[ijk   ])/tPr;
    //                 eviscn = TF(0.5)*(evisc[ijk   ]+evisc[ijk+jj])/tPr;
    //                 eviscs = TF(0.5)*(evisc[ijk-jj]+evisc[ijk   ])/tPr;
    //                 evisct = TF(0.5)*(evisc[ijk   ]+evisc[ijk+kk])/tPr;
    //                 eviscb = TF(0.5)*(evisc[ijk-kk]+evisc[ijk   ])/tPr;
    //
    //                 at[ijk] +=
    //                          + ( evisce*(a[ijk+ii]-a[ijk   ])
    //                            - eviscw*(a[ijk   ]-a[ijk-ii]) ) * dxidxi
    //                          + ( eviscn*(a[ijk+jj]-a[ijk   ])
    //                            - eviscs*(a[ijk   ]-a[ijk-jj]) ) * dyidyi
    //                          + ( rhorefh[kstart+1] * evisct*(a[ijk+kk]-a[ijk   ])*dzhi[kstart+1]
    //                            + rhorefh[kstart  ] * fluxbot[ij] ) / rhoref[kstart] * dzi[kstart];
    //             }
    //
    //         // top boundary
    //         for (int j=jstart; j<jend; ++j)
    //             #pragma ivdep
    //             for (int i=istart; i<iend; ++i)
    //             {
    //                 const int ij  = i + j*jj;
    //                 const int ijk = i + j*jj + (kend-1)*kk;
    //                 evisce = TF(0.5)*(evisc[ijk   ]+evisc[ijk+ii])/tPr;
    //                 eviscw = TF(0.5)*(evisc[ijk-ii]+evisc[ijk   ])/tPr;
    //                 eviscn = TF(0.5)*(evisc[ijk   ]+evisc[ijk+jj])/tPr;
    //                 eviscs = TF(0.5)*(evisc[ijk-jj]+evisc[ijk   ])/tPr;
    //                 evisct = TF(0.5)*(evisc[ijk   ]+evisc[ijk+kk])/tPr;
    //                 eviscb = TF(0.5)*(evisc[ijk-kk]+evisc[ijk   ])/tPr;
    //
    //                 at[ijk] +=
    //                          + ( evisce*(a[ijk+ii]-a[ijk   ])
    //                            - eviscw*(a[ijk   ]-a[ijk-ii]) ) * dxidxi
    //                          + ( eviscn*(a[ijk+jj]-a[ijk   ])
    //                            - eviscs*(a[ijk   ]-a[ijk-jj]) ) * dyidyi
    //                          + (-rhorefh[kend  ] * fluxtop[ij]
    //                            - rhorefh[kend-1] * eviscb*(a[ijk   ]-a[ijk-kk])*dzhi[kend-1] ) / rhoref[kend-1] * dzi[kend-1];
    //             }
    //     }
    //
    //     for (int k=kstart+k_offset; k<kend-k_offset; ++k)
    //         for (int j=jstart; j<jend; ++j)
    //             #pragma ivdep
    //             for (int i=istart; i<iend; ++i)
    //             {
    //                 const int ijk = i + j*jj + k*kk;
    //                 evisce = TF(0.5)*(evisc[ijk   ]+evisc[ijk+ii])/tPr;
    //                 eviscw = TF(0.5)*(evisc[ijk-ii]+evisc[ijk   ])/tPr;
    //                 eviscn = TF(0.5)*(evisc[ijk   ]+evisc[ijk+jj])/tPr;
    //                 eviscs = TF(0.5)*(evisc[ijk-jj]+evisc[ijk   ])/tPr;
    //                 evisct = TF(0.5)*(evisc[ijk   ]+evisc[ijk+kk])/tPr;
    //                 eviscb = TF(0.5)*(evisc[ijk-kk]+evisc[ijk   ])/tPr;
    //
    //                 at[ijk] +=
    //                          + ( evisce*(a[ijk+ii]-a[ijk   ])
    //                            - eviscw*(a[ijk   ]-a[ijk-ii]) ) * dxidxi
    //                          + ( eviscn*(a[ijk+jj]-a[ijk   ])
    //                            - eviscs*(a[ijk   ]-a[ijk-jj]) ) * dyidyi
    //                          + ( rhorefh[k+1] * evisct*(a[ijk+kk]-a[ijk   ])*dzhi[k+1]
    //                            - rhorefh[k  ] * eviscb*(a[ijk   ]-a[ijk-kk])*dzhi[k]  ) / rhoref[k] * dzi[k];
    //             }
    // }

} // End namespace.


{ // tijdelijke hulphaakjes, verwijder later
  template<typename TF>
  Diff_tke2<TF>::Diff_tke2(Master& masterin, Grid<TF>& gridin, Fields<TF>& fieldsin, Input& inputin) :
      Diff<TF>(masterin, gridin, fieldsin, inputin),
      boundary_cyclic(master, grid),
      field3d_operators(master, grid, fields)
  {
      dnmax = inputin.get_item<TF>("diff", "dnmax", "", 0.4  );
      cs    = inputin.get_item<TF>("diff", "cs"   , "", 0.23 );
      //tPr   = inputin.get_item<TF>("diff", "tPr"  , "", 1./3.);

      fields.init_prognostic_field("sgs_tke", "Subgrid scale TKE", "m2 s-2");
      fields.init_diagnostic_field("evisc", "Eddy viscosity", "m2 s-1");
  }

  template<typename TF>
  Diff_tke2<TF>::~Diff_tke2()
  {
  }

  template<typename TF>
  void Diff_tke2<TF>::init()
  {
      boundary_cyclic.init();
      // NOOT: Moet hier ook de init_stat komen?
  }

  template<typename TF>
  Diffusion_type Diff_tke2<TF>::get_switch() const
  {
      return swdiff;
  }

  template<typename TF>
  unsigned long Diff_tke2<TF>::get_time_limit(const unsigned long idt, const double dt)
  {
      // Ugly solution for now, to avoid passing entire tPr-field, or recalculate it again; SvdLinden, June 2018
      double Pr_max = 2.0;

      auto& gd = grid.get_grid_data();
      double dnmul = calc_dnmul<TF>(fields.sd["evisc"]->fld.data(), gd.dzi.data(), 1./(gd.dx*gd.dx), 1./(gd.dy*gd.dy), Pr_max,
                                    gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                                    gd.icells, gd.ijcells);
      master.max(&dnmul, 1);

      // Avoid zero division.
      dnmul = std::max(Constants::dsmall, dnmul);

      return idt * dnmax / (dt * dnmul);
  }

  template<typename TF>
  double Diff_tke2<TF>::get_dn(const double dt)
  {
      // Ugly solution for now, to avoid passing entire tPr-field, or recalculate it again; SvdLinden, June 2018
      double Pr_max = 2.0;

      auto& gd = grid.get_grid_data();
      double dnmul = calc_dnmul<TF>(fields.sd["evisc"]->fld.data(), gd.dzi.data(), 1./(gd.dx*gd.dx), 1./(gd.dy*gd.dy), Pr_max,
                                    gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                                    gd.icells, gd.ijcells);
      master.max(&dnmul, 1);

      return dnmul*dt;
  }

  // Deze snap ik in zijn geheel niet, waarom doet de standaard viscositeit hier er toe??
  template<typename TF>
  void Diff_tke2<TF>::set_values()
  {
      auto& gd = grid.get_grid_data();

      // Get the maximum viscosity
      TF viscmax = fields.visc;
      for (auto& it : fields.sp)
          viscmax = std::max(it.second->visc, viscmax);

      // Calculate time step multiplier for diffusion number
      dnmul = 0;
      for (int k=gd.kstart; k<gd.kend; ++k)
          dnmul = std::max(dnmul, std::abs(viscmax * (1./(gd.dx*gd.dx) + 1./(gd.dy*gd.dy) + 1./(gd.dz[k]*gd.dz[k]))));
  }

  #ifndef USECUDA
  template<typename TF>
  void Diff_tke2<TF>::exec(Boundary<TF>& boundary)
  {
      auto& gd = grid.get_grid_data();

      if (boundary.get_switch() == "surface")
      {
          diff_u<TF, Surface_model::Enabled>(
                  fields.mt["u"]->fld.data(),
                  fields.mp["u"]->fld.data(), fields.mp["v"]->fld.data(), fields.mp["w"]->fld.data(),
                  gd.dzi.data(), gd.dzhi.data(), 1./gd.dx, 1./gd.dy,
                  fields.sd["evisc"]->fld.data(),
                  fields.mp["u"]->flux_bot.data(), fields.mp["u"]->flux_top.data(),
                  fields.rhoref.data(), fields.rhorefh.data(),
                  gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                  gd.icells, gd.ijcells);

          diff_v<TF, Surface_model::Enabled>(
                  fields.mt["v"]->fld.data(),
                  fields.mp["u"]->fld.data(), fields.mp["v"]->fld.data(), fields.mp["w"]->fld.data(),
                  gd.dzi.data(), gd.dzhi.data(), 1./gd.dx, 1./gd.dy,
                  fields.sd["evisc"]->fld.data(),
                  fields.mp["v"]->flux_bot.data(), fields.mp["v"]->flux_top.data(),
                  fields.rhoref.data(), fields.rhorefh.data(),
                  gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                  gd.icells, gd.ijcells);

          diff_w<TF>(fields.mt["w"]->fld.data(),
                     fields.mp["u"]->fld.data(), fields.mp["v"]->fld.data(), fields.mp["w"]->fld.data(),
                     gd.dzi.data(), gd.dzhi.data(), 1./gd.dx, 1./gd.dy,
                     fields.sd["evisc"]->fld.data(),
                     fields.rhoref.data(), fields.rhorefh.data(),
                     gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                     gd.icells, gd.ijcells);

          for (auto it : fields.st)
              diff_c<TF, Surface_model::Enabled>(
                      it.second->fld.data(), fields.sp[it.first]->fld.data(),
                      gd.dzi.data(), gd.dzhi.data(), 1./(gd.dx*gd.dx), 1./(gd.dy*gd.dy),
                      fields.sd["evisc"]->fld.data(),
                      fields.sp[it.first]->flux_bot.data(), fields.sp[it.first]->flux_top.data(),
                      fields.rhoref.data(), fields.rhorefh.data(), tPr,
                      gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                      gd.icells, gd.ijcells);
      }
      else
      {
          diff_u<TF, Surface_model::Disabled>(
                  fields.mt["u"]->fld.data(),
                  fields.mp["u"]->fld.data(), fields.mp["v"]->fld.data(), fields.mp["w"]->fld.data(),
                  gd.dzi.data(), gd.dzhi.data(), 1./gd.dx, 1./gd.dy,
                  fields.sd["evisc"]->fld.data(),
                  fields.mp["u"]->flux_bot.data(), fields.mp["u"]->flux_top.data(),
                  fields.rhoref.data(), fields.rhorefh.data(),
                  gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                  gd.icells, gd.ijcells);

          diff_v<TF, Surface_model::Disabled>(
                  fields.mt["v"]->fld.data(),
                  fields.mp["u"]->fld.data(), fields.mp["v"]->fld.data(), fields.mp["w"]->fld.data(),
                  gd.dzi.data(), gd.dzhi.data(), 1./gd.dx, 1./gd.dy,
                  fields.sd["evisc"]->fld.data(),
                  fields.mp["v"]->flux_bot.data(), fields.mp["v"]->flux_top.data(),
                  fields.rhoref.data(), fields.rhorefh.data(),
                  gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                  gd.icells, gd.ijcells);

          diff_w<TF>(fields.mt["w"]->fld.data(),
                     fields.mp["u"]->fld.data(), fields.mp["v"]->fld.data(), fields.mp["w"]->fld.data(),
                     gd.dzi.data(), gd.dzhi.data(), 1./gd.dx, 1./gd.dy,
                     fields.sd["evisc"]->fld.data(),
                     fields.rhoref.data(), fields.rhorefh.data(),
                     gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                     gd.icells, gd.ijcells);

          for (auto it : fields.st)
              diff_c<TF, Surface_model::Disabled>(
                      it.second->fld.data(), fields.sp[it.first]->fld.data(),
                      gd.dzi.data(), gd.dzhi.data(), 1./(gd.dx*gd.dx), 1./(gd.dy*gd.dy),
                      fields.sd["evisc"]->fld.data(),
                      fields.sp[it.first]->flux_bot.data(), fields.sp[it.first]->flux_top.data(),
                      fields.rhoref.data(), fields.rhorefh.data(), tPr,
                      gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                      gd.icells, gd.ijcells);
      }
  }

  template<typename TF>
  void Diff_tke2<TF>::exec_viscosity(Boundary<TF>& boundary, Thermo<TF>& thermo)
  {
      auto& gd = grid.get_grid_data();

      // Calculate strain rate using MO for velocity gradients lowest level.
      if (boundary.get_switch() == "surface")
          calc_strain2<TF, Surface_model::Enabled>(
                  fields.sd["evisc"]->fld.data(),
                  fields.mp["u"]->fld.data(), fields.mp["v"]->fld.data(), fields.mp["w"]->fld.data(),
                  fields.mp["u"]->flux_bot.data(), fields.mp["v"]->flux_bot.data(),
                  boundary.ustar.data(), boundary.obuk.data(),
                  gd.z.data(), gd.dzi.data(), gd.dzhi.data(), 1./gd.dx, 1./gd.dy,
                  gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                  gd.icells, gd.ijcells);

      // Calculate strain rate using resolved boundaries.
      else
          calc_strain2<TF, Surface_model::Disabled>(
                  fields.sd["evisc"]->fld.data(),
                  fields.mp["u"]->fld.data(), fields.mp["v"]->fld.data(), fields.mp["w"]->fld.data(),
                  fields.mp["u"]->flux_bot.data(), fields.mp["v"]->flux_bot.data(),
                  nullptr, nullptr,
                  gd.z.data(), gd.dzi.data(), gd.dzhi.data(), 1./gd.dx, 1./gd.dy,
                  gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                  gd.icells, gd.ijcells);

      // Start with retrieving the stability information
      if (thermo.get_switch() == "0")
      {
           // Calculate eddy viscosity using MO at lowest model level
          if (boundary.get_switch() == "surface")
              calc_evisc_neutral<TF, Surface_model::Enabled>(
                      fields.sd["evisc"]->fld.data(),
                      fields.mp["u"]->fld.data(), fields.mp["v"]->fld.data(), fields.mp["w"]->fld.data(),
                      fields.mp["u"]->flux_bot.data(), fields.mp["v"]->flux_bot.data(),
                      gd.z.data(), gd.dz.data(), boundary.z0m, fields.visc,
                      gd.dx, gd.dy, this->cs,
                      gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                      gd.icells, gd.jcells, gd.ijcells,
                      boundary_cyclic);

           // Calculate eddy viscosity assuming resolved walls
          else
              calc_evisc_neutral<TF, Surface_model::Disabled>(
                      fields.sd["evisc"]->fld.data(),
                      fields.mp["u"]->fld.data(), fields.mp["v"]->fld.data(), fields.mp["w"]->fld.data(),
                      fields.mp["u"]->flux_bot.data(), fields.mp["v"]->flux_bot.data(),
                      gd.z.data(), gd.dz.data(), boundary.z0m, fields.visc,
                      gd.dx, gd.dy, this->cs,
                      gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                      gd.icells, gd.jcells, gd.ijcells,
                      boundary_cyclic);
      }
      // assume buoyancy calculation is needed
      else
      {
          // store the buoyancyflux in tmp1
          auto& gd = grid.get_grid_data();
          auto buoy_tmp = fields.get_tmp();
          auto tmp = fields.get_tmp();
          thermo.get_buoyancy_fluxbot(*buoy_tmp, false);
          thermo.get_thermo_field(*buoy_tmp, "N2", false, false);

          if (boundary.get_switch() == "surface")
              calc_evisc<TF, Surface_model::Enabled>(
                      fields.sd["evisc"]->fld.data(),
                      fields.mp["u"]->fld.data(), fields.mp["v"]->fld.data(), fields.mp["w"]->fld.data(), buoy_tmp->fld.data(),
                      fields.mp["u"]->flux_bot.data(), fields.mp["v"]->flux_bot.data(), buoy_tmp->flux_bot.data(),
                      boundary.ustar.data(), boundary.obuk.data(),
                      gd.z.data(), gd.dz.data(), gd.dzi.data(),
                      gd.dx, gd.dy,
                      boundary.z0m, fields.visc, this->cs, this->tPr,
                      gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                      gd.icells, gd.jcells, gd.ijcells,
                      boundary_cyclic);
          else
              calc_evisc<TF, Surface_model::Disabled>(
                      fields.sd["evisc"]->fld.data(),
                      fields.mp["u"]->fld.data(), fields.mp["v"]->fld.data(), fields.mp["w"]->fld.data(), buoy_tmp->fld.data(),
                      fields.mp["u"]->flux_bot.data(), fields.mp["v"]->flux_bot.data(), buoy_tmp->flux_bot.data(),
                      nullptr, nullptr,
                      gd.z.data(), gd.dz.data(), gd.dzi.data(),
                      gd.dx, gd.dy,
                      boundary.z0m, fields.visc, this->cs, this->tPr,
                      gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                      gd.icells, gd.jcells, gd.ijcells,
                      boundary_cyclic);

          fields.release_tmp(buoy_tmp);
          fields.release_tmp(tmp);
      }
  }
  #endif
} // tijdelijke hulphaakjes, verwijder later
template class Diff_tke2<double>;
template class Diff_tke2<float>;
