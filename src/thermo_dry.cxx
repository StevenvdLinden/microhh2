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

#include <cstdio>
#include <algorithm>
#include <cmath>
#include <sstream>
#include "grid.h"
#include "fields.h"
#include "defines.h"
#include "constants.h"
#include "finite_difference.h"
#include "data_block.h"
#include "stats.h"
#include "diff_smag2.h"
#include "master.h"
#include "cross.h"
#include "column.h"
#include "dump.h"

#include "thermo_dry.h"
#include "thermo_moist_functions.h"  // For Exner function

using Finite_difference::O4::interp4c;
using namespace Constants;
using namespace Thermo_moist_functions;

namespace
{
    template<typename TF>
    void calc_buoyancy(TF* const restrict b, const TF* const restrict th, const TF* const restrict thref,
                       const int istart, const int iend, const int jstart, const int jend,
                       const int icells, const int ijcells, const int kcells )
    {
        for (int k=0; k<kcells; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*icells + k*ijcells;
                    b[ijk] = grav<TF>/thref[k] * (th[ijk] - thref[k]);
                }
    }

    template<typename TF>
    void calc_N2(TF* const restrict N2, const TF* const restrict th, const TF* const restrict dzi, const TF* const restrict thref,
                 const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
                 const int icells, const int ijcells, const int kcells)
    {
        for (int k=kstart; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*icells + k*ijcells;
                    N2[ijk] = grav<TF>/thref[k]*TF(0.5)*(th[ijk+ijcells] - th[ijk-ijcells])*dzi[k];
                }
    }

    template<typename TF>
    void calc_T(TF* const restrict T, const TF* const restrict th,
                const TF* const restrict exnref, const TF* const restrict thref,
                const int istart, const int iend, const int jstart, const int jend,
                const int jj, const int kk, const int kcells)
    {
        for (int k=0; k<kcells; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj+ k*kk;
                    T[ijk] = exnref[k]*thref[k] + (th[ijk]-thref[k]);
                }
    }

    template<typename TF>
    void calc_T_h(TF* const restrict T, const TF* const restrict th,
                  const TF* const restrict exnrefh, const TF* const restrict threfh,
                  const int istart, const int iend, const int jstart, const int jend,
                  const int jj, const int kk, const int kcells)
    {
        using Finite_difference::O2::interp2;

        for (int k=0; k<kcells; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj+ k*kk;
                    T[ijk] = exnrefh[k]*threfh[k] + (interp2(th[ijk-kk], th[ijk]) - threfh[k]);
                }
    }

    template<typename TF>
    void calc_T_bot(TF* const restrict T_bot, const TF* const restrict th,
                    const TF* const restrict exnrefh, const TF* const restrict threfh,
                    const int istart, const int iend, const int jstart, const int jend, const int kstart,
                    const int jj, const int kk)
    {
        using Finite_difference::O2::interp2;

        for (int j=jstart; j<jend; ++j)
            #pragma ivdep
            for (int i=istart; i<iend; ++i)
            {
                const int ij = i + j*jj;
                const int ijk = i + j*jj + kstart*kk;
                T_bot[ij] = exnrefh[kstart]*threfh[kstart] + (interp2(th[ijk-kk], th[ijk]) - threfh[kstart]);
            }
    }

    template<typename TF>
    void calc_buoyancy_bot(TF* const restrict b , TF* const restrict bbot,
                           const TF* const restrict th, const TF* const restrict thbot,
                           const TF* const restrict thref, const TF* const restrict threfh,
                           const int icells, const int jcells, const int kstart,
                           const int ijcells)
    {
        for (int j=0; j<jcells; ++j)
            #pragma ivdep
            for (int i=0; i<icells; ++i)
            {
                const int ij  = i + j*icells;
                const int ijk = i + j*icells + kstart*ijcells;
                bbot[ij] = grav<TF>/threfh[kstart] * (thbot[ij] - threfh[kstart]);
                b[ijk]   = grav<TF>/thref [kstart] * (th[ijk]   - thref [kstart]);
            }
    }

    template<typename TF>
    void calc_buoyancy_fluxbot(TF* const restrict bfluxbot, const TF* const restrict thfluxbot,const TF* const restrict threfh,
                                       const int icells, const int jcells, const int kstart,
                                       const int ijcells)
    {
        for (int j=0; j<jcells; ++j)
            #pragma ivdep
            for (int i=0; i<icells; ++i)
            {
                const int ij = i + j*icells;
                bfluxbot[ij] = grav<TF>/threfh[kstart]*thfluxbot[ij];
            }
    }

    template<typename TF>
    void calc_buoyancy_tend_2nd(TF* const restrict wt, const TF* const restrict th, const TF* const restrict threfh,
                                const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
                                const int icells, const int ijcells)
    {
        using Finite_difference::O2::interp2;

        for (int k=kstart+1; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*icells + k*ijcells;
                    wt[ijk] += grav<TF>/threfh[k] * (interp2(th[ijk-ijcells], th[ijk]) - threfh[k]);
                }
    }

    template<typename TF>
    void calc_buoyancy_tend_4th(TF* const restrict wt, const TF* const restrict th, const TF* const restrict threfh,
                                const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
                                const int icells, const int ijcells)
    {
        using Finite_difference::O4::interp4c;

        const int ijcells2 = 2*ijcells;
        for (int k=kstart+1; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*icells + k*ijcells;
                    wt[ijk] += grav<TF>/threfh[k] * (interp4c(th[ijk-ijcells2], th[ijk-ijcells], th[ijk], th[ijk+ijcells]) - threfh[k]);
                }
    }

    template<typename TF>
    void calc_baroclinic(TF* const restrict tht, const TF* const restrict v,
                         const TF dthetady_ls,
                         const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
                         const int jj, const int kk)
    {
        using Finite_difference::O2::interp2;

        for (int k=kstart; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    tht[ijk] -= dthetady_ls * interp2(v[ijk], v[ijk+jj]);
                }
    }

    // Initialize the base state for the anelastic solver
    template<typename TF>
    void calc_base_state(TF* const restrict rhoref, TF* const restrict rhorefh,
                         TF* const restrict pref,   TF* const restrict prefh,
                         TF* const restrict exnref, TF* const restrict exnrefh,
                         TF* const restrict thref,  TF* const restrict threfh,
                         const TF pbot, const TF* const restrict z, const TF* const restrict zh,
                         const TF* const restrict dz, const TF* const restrict dzh, const TF* const restrict dzhi,
                         const int kstart, const int kend, const int kcells)
    {
        // extrapolate the input sounding to get the bottom value
        threfh[kstart] = thref[kstart] - z[kstart]*(thref[kstart+1]-thref[kstart])*dzhi[kstart+1];

        // extrapolate the input sounding to get the top value
        threfh[kend] = thref[kend-1] + (zh[kend]-z[kend-1])*(thref[kend-1]-thref[kend-2])*dzhi[kend-1];

        // set the ghost cells for the reference potential temperature
        thref[kstart-1] = TF(2.)*threfh[kstart] - thref[kstart];
        thref[kend]     = TF(2.)*threfh[kend]   - thref[kend-1];

        // interpolate the input sounding to half levels
        for (int k=kstart+1; k<kend; ++k)
            threfh[k] = TF(0.5)*(thref[k-1] + thref[k]);

        // Calculate pressure
        prefh[kstart] = pbot;
        pref [kstart] = pbot * std::exp(-grav<TF> * z[kstart] / (Rd<TF> * threfh[kstart] * exner(prefh[kstart])));
        for (int k=kstart+1; k<kend+1; ++k)
        {
            prefh[k] = prefh[k-1] * std::exp(-grav<TF> * dz[k-1] / (Rd<TF> * thref[k-1] * exner(pref[k-1])));
            pref [k] = pref [k-1] * std::exp(-grav<TF> * dzh[k ] / (Rd<TF> * threfh[k ] * exner(prefh[k ])));
        }
        pref[kstart-1] = TF(2.)*prefh[kstart] - pref[kstart];

        // Calculate density and exner
        for (int k=0; k<kcells; ++k)
        {
            exnref[k]  = exner(pref[k] );
            exnrefh[k] = exner(prefh[k]);
            rhoref[k]  = pref[k]  / (Rd<TF> * thref[k]  * exnref[k] );
            rhorefh[k] = prefh[k] / (Rd<TF> * threfh[k] * exnrefh[k]);
        }
    }
}

template<typename TF>
Thermo_dry<TF>::Thermo_dry(Master& masterin, Grid<TF>& gridin, Fields<TF>& fieldsin, Input& inputin) :
Thermo<TF>(masterin, gridin, fieldsin, inputin),
boundary_cyclic(masterin, gridin)
{
    swthermo = "dry";

    fields.init_prognostic_field("th", "Potential Temperature", "K");

    fields.sp.at("th")->visc = inputin.get_item<TF>("fields", "svisc", "th");

    std::string swbasestate_in = inputin.get_item<std::string>("thermo", "swbasestate", "", "");
    if (swbasestate_in == "boussinesq")
        bs.swbasestate = Basestate_type::boussinesq;
    else if (swbasestate_in == "anelastic")
        bs.swbasestate = Basestate_type::anelastic;
    else
        throw std::runtime_error("Invalid option for \"swbasestate\"");

    if (grid.get_spatial_order() == Grid_order::Fourth && bs.swbasestate == Basestate_type::anelastic)
    {
        master.print_error("Anelastic mode is not supported for swspatialorder=4\n");
        throw std::runtime_error("Illegal options swbasestate");
    }

    swbaroclinic = inputin.get_item<bool>("thermo", "swbaroclinic", "", false);
    if (swbaroclinic)
        dthetady_ls = inputin.get_item<TF>("thermo", "dthetady_ls", "");

}

template<typename TF>
Thermo_dry<TF>::~Thermo_dry()
{
}

template<typename TF>
void Thermo_dry<TF>::init()
{
    auto& gd = grid.get_grid_data();

    bs.thref.resize(gd.kcells);
    bs.pref.resize(gd.kcells);
    bs.threfh.resize(gd.kcells);
    bs.prefh.resize(gd.kcells);
    bs.exnref.resize(gd.kcells);
    bs.exnrefh.resize(gd.kcells);
}

template<typename TF>
void Thermo_dry<TF>::create(Input& inputin, Data_block& data_block, Stats<TF>& stats, Column<TF>& column, Cross<TF>& cross, Dump<TF>& dump)
{
    auto& gd = grid.get_grid_data();
    /* Setup base state:
       For anelastic setup, calculate reference density and temperature from input sounding
       For boussinesq, reference density and temperature are fixed */

    if (bs.swbasestate == Basestate_type::anelastic)
    {
        bs.pbot = inputin.get_item<TF>("thermo", "pbot", "");

        // Read the reference profile, and start writing it at index kstart as thref is kcells long.
        data_block.get_vector(bs.thref, "th", gd.kmax, 0, gd.kstart);

        calc_base_state(
                fields.rhoref.data(), fields.rhorefh.data(), bs.pref.data(), bs.prefh.data(),
                bs.exnref.data(), bs.exnrefh.data(), bs.thref.data(), bs.threfh.data(), bs.pbot,
                gd.z.data(), gd.zh.data(), gd.dz.data(), gd.dzh.data(), gd.dzhi.data(), gd.kstart, gd.kend, gd.kcells);
    }
    else
    {
        bs.thref0 = inputin.get_item<TF>("thermo", "thref0", "");

        // Set entire column to reference value. Density is already initialized at 1.0 in fields.cxx.
        for (int k=0; k<gd.kcells; ++k)
        {
            bs.thref[k]  = bs.thref0;
            bs.threfh[k] = bs.thref0;
        }
    }

    // Init the toolbox classes.
    boundary_cyclic.init();

    // Set up output classes
    create_stats(stats);
    create_column(column);
    create_dump(dump);
    create_cross(cross);
}

#ifndef USECUDA
template<typename TF>
void Thermo_dry<TF>::exec( const double dt)
{
    auto& gd = grid.get_grid_data();

    if (grid.get_spatial_order() == Grid_order::Second)
        calc_buoyancy_tend_2nd(fields.mt.at("w")->fld.data(), fields.sp.at("th")->fld.data(), bs.threfh.data(),
                               gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                               gd.icells, gd.ijcells);
    else if (grid.get_spatial_order() == Grid_order::Fourth)
        calc_buoyancy_tend_4th(fields.mt.at("w")->fld.data(), fields.sp.at("th")->fld.data(), bs.threfh.data(),
                               gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                               gd.icells, gd.ijcells);

    if (swbaroclinic)
        calc_baroclinic(fields.st.at("th")->fld.data(), fields.mp.at("v")->fld.data(),
                        dthetady_ls,
                        gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                        gd.icells, gd.ijcells);
}
#endif

template<typename TF>
unsigned long Thermo_dry<TF>::get_time_limit(unsigned long idt, const double dt)
{
    return Constants::ulhuge;
}

template<typename TF>
bool Thermo_dry<TF>::check_field_exists(std::string name)
{
    if (name == "b" || name == "T")
        return true;
    else
        return false;
}

template<typename TF>
void Thermo_dry<TF>::get_thermo_field(Field3d<TF>& fld, std::string name, bool cyclic, bool is_stat)
{
    auto& gd = grid.get_grid_data();
    background_state base;
    if (is_stat)
        base = bs_stats;
    else
        base = bs;


    if (name == "b")
        calc_buoyancy(fld.fld.data(), fields.sp.at("th")->fld.data(), base.thref.data(),
                      gd.istart, gd.iend, gd.jstart, gd.jend, gd.icells, gd.ijcells, gd.kcells);
    else if (name == "N2")
        calc_N2(fld.fld.data(), fields.sp.at("th")->fld.data(), gd.dzi.data(), base.thref.data(),
                gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend, gd.icells, gd.ijcells, gd.kcells);
    else if (name == "T")
        calc_T(fld.fld.data(), fields.sp.at("th")->fld.data(), base.exnref.data(), base.thref.data(),
               gd.istart, gd.iend, gd.jstart, gd.jend, gd.icells, gd.ijcells, gd.kcells);
    else if (name == "T_h")
        calc_T_h(fld.fld.data(), fields.sp.at("th")->fld.data(), base.exnrefh.data(), base.threfh.data(),
                 gd.istart, gd.iend, gd.jstart, gd.jend, gd.icells, gd.ijcells, gd.kcells);
    else
    {
        throw std::runtime_error("Illegal thermo field");
    }

    if (cyclic)
        boundary_cyclic.exec(fld.fld.data());
}

template<typename TF>
void Thermo_dry<TF>::get_buoyancy_fluxbot(Field3d<TF>& b, bool is_stat)
{
    auto& gd = grid.get_grid_data();
    background_state base;
    if (is_stat)
        base = bs_stats;
    else
        base = bs;


    calc_buoyancy_fluxbot(b.flux_bot.data(), fields.sp.at("th")->flux_bot.data(), base.threfh.data(),
                          gd.icells, gd.jcells, gd.kstart, gd.ijcells);
}

template<typename TF>
void Thermo_dry<TF>::get_buoyancy_surf(Field3d<TF>& b, bool is_stat)
{
    auto& gd = grid.get_grid_data();
    background_state base;
    if (is_stat)
        base = bs_stats;
    else
        base = bs;


    calc_buoyancy_bot(b.fld.data(), b.fld_bot.data(), fields.sp.at("th")->fld.data(), fields.sp.at("th")->fld_bot.data(),
                      base.thref.data(), base.threfh.data(),
                      gd.icells, gd.jcells, gd.kstart, gd.ijcells);
    calc_buoyancy_fluxbot(b.flux_bot.data(), fields.sp.at("th")->flux_bot.data(), base.threfh.data(),
                          gd.icells, gd.jcells, gd.kstart, gd.ijcells);
}

template<typename TF>
void Thermo_dry<TF>::get_T_bot(Field3d<TF>& T_bot, bool is_stat)
{
    auto& gd = grid.get_grid_data();
    background_state base;
    if (is_stat)
        base = bs_stats;
    else
        base = bs;

    calc_T_bot(T_bot.fld_bot.data(), fields.sp.at("th")->fld.data(), base.exnrefh.data(), base.threfh.data(),
               gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.icells, gd.ijcells);
}

template<typename TF>
const std::vector<TF>& Thermo_dry<TF>::get_p_vector() const
{
    return bs.pref;
}

template<typename TF>
const std::vector<TF>& Thermo_dry<TF>::get_ph_vector() const
{
    return bs.prefh;
}

template<typename TF>
const std::vector<TF>& Thermo_dry<TF>::get_exner_vector() const
{
    return bs.exnref;
}

template<typename TF>
void Thermo_dry<TF>::get_prog_vars(std::vector<std::string>& list)
{
    list.push_back("th");
}

template<typename TF>
TF Thermo_dry<TF>::get_buoyancy_diffusivity()
{
    // Use the diffusivity from theta
    return fields.sp.at("th")->visc;
}

template <typename TF>
void Thermo_dry<TF>::create_stats(Stats<TF>& stats)
{
    // Add the profiles to te statistics
    if (stats.get_switch())
    {
        bs_stats = bs;
        // Add base state profiles to statistics
        stats.add_fixed_prof("rhoref",  "Full level basic state density",  "kg m-3", "z",  fields.rhoref.data());
        stats.add_fixed_prof("rhorefh", "Half level basic state density",  "kg m-3", "zh", fields.rhorefh.data());
        stats.add_fixed_prof("thref",   "Full level basic state potential temperature", "K", "z", bs_stats.thref.data());
        stats.add_fixed_prof("threfh",  "Half level basic state potential temperature", "K", "zh",bs_stats.thref.data());
        if (bs_stats.swbasestate == Basestate_type::anelastic)
        {
            stats.add_fixed_prof("ph",  "Full level hydrostatic pressure", "Pa", "z",  bs_stats.pref.data());
            stats.add_fixed_prof("phh", "Half level hydrostatic pressure", "Pa", "zh", bs_stats.prefh.data());
            stats.add_prof("T", "Absolute temperature", "K", "z");
        }

        stats.add_prof("b", "Buoyancy", "m s-2", "z");
        for (int n=2; n<5; ++n)
        {
            std::string sn = std::to_string(n);
            stats.add_prof("b"+sn, "Moment " +sn+" of the buoyancy", "(m s-2)"+sn,"z");
        }

        stats.add_prof("bgrad", "Gradient of the buoyancy", "s-2", "zh");
        stats.add_prof("bw"   , "Turbulent flux of the buoyancy", "m2 s-3", "zh");
        stats.add_prof("bdiff", "Diffusive flux of the buoyancy", "m2 s-3", "zh");
        stats.add_prof("bflux", "Total flux of the buoyancy", "m2 s-3", "zh");

        // stats.add_prof("bsort", "Sorted buoyancy", "m s-2", "z");
    }
}

template<typename TF>
void Thermo_dry<TF>::create_column(Column<TF>& column)
{
    // add the profiles to the columns
    if (column.get_switch())
        column.add_prof("b", "Buoyancy", "m s-2", "z");
}

template<typename TF>
void Thermo_dry<TF>::create_dump(Dump<TF>& dump)
{
    // add the profiles to the columns
    if (dump.get_switch())
    {
        // Get global dump-list from dump.cxx
        std::vector<std::string> *dumplist_global = dump.get_dumplist();

        // Check if fields in dumplist are diagnostic fields, if not delete them and print warning
        std::vector<std::string>::iterator dumpvar=dumplist_global->begin();
        while (dumpvar != dumplist_global->end())
        {
            if (check_field_exists(*dumpvar))
            {
                // Remove variable from global list, put in local list
                dumplist.push_back(*dumpvar);
                dumplist_global->erase(dumpvar); // erase() returns iterator of next element..
            }
            else
                ++dumpvar;
        }
    }
}

template<typename TF>
void Thermo_dry<TF>::create_cross(Cross<TF>& cross)
{
    if (cross.get_switch())
    {
        swcross_b = false;

        // Populate list with allowed cross-section variables
        allowedcrossvars.push_back("b");
        allowedcrossvars.push_back("bbot");
        allowedcrossvars.push_back("bfluxbot");
        allowedcrossvars.push_back("blngrad");
        allowedcrossvars.push_back("thlngrad");

        // Get global cross-list from cross.cxx
        std::vector<std::string> *crosslist_global = cross.get_crosslist();

        // Check input list of cross variables (crosslist)
        std::vector<std::string>::iterator it=crosslist_global->begin();
        while (it != crosslist_global->end())
        {
            if (std::count(allowedcrossvars.begin(), allowedcrossvars.end(), *it))
            {
                // Remove variable from global list, put in local list
                crosslist.push_back(*it);
                crosslist_global->erase(it); // erase() returns iterator of next element..

                // CvH: This is not required if only thlngrad is used.
                swcross_b = true;
            }
            else
                ++it;
        }
   }
}

template<typename TF>
void Thermo_dry<TF>::exec_stats(Stats<TF>& stats, std::string mask_name, Field3d<TF>& mask_field, Field3d<TF>& mask_fieldh,
        const Diff<TF>& diff, const double dt)
{
    auto& gd = grid.get_grid_data();
    #ifndef USECUDA
        bs_stats = bs;
    #endif

    Mask<TF>& m = stats.masks[mask_name];

    const TF no_offset = 0.;

    // calculate the buoyancy and its surface flux for the profiles
    auto b = fields.get_tmp();
    get_thermo_field(*b, "b",true, true);
    get_buoyancy_surf(*b, true);
    get_buoyancy_fluxbot(*b, true);


    // define the location
    const int sloc[] = {0,0,0};

    // calculate the mean
    stats.calc_mean(m.profs["b"].data.data(), b->fld.data(), no_offset, mask_field.fld.data(), stats.nmask.data());

    // calculate the moments
    for (int n=2; n<5; ++n)
    {
        std::string sn = std::to_string(n);
        stats.calc_moment(b->fld.data(), m.profs["b"].data.data(), m.profs["b"+sn].data.data(), n, mask_field.fld.data(), stats.nmask.data());
    }

    if (grid.get_spatial_order() == Grid_order::Second)
    {
        auto tmp = fields.get_tmp();
        stats.calc_grad_2nd(b->fld.data(), m.profs["bgrad"].data.data(), gd.dzhi.data(), mask_fieldh.fld.data(), stats.nmaskh.data());
        stats.calc_flux_2nd(b->fld.data(), m.profs["b"].data.data(), fields.mp["w"]->fld.data(), m.profs["w"].data.data(),
                            m.profs["bw"].data.data(), tmp->fld.data(), sloc, mask_fieldh.fld.data(), stats.nmaskh.data());
        if (diff.get_switch() == Diffusion_type::Diff_smag2)
        {
            stats.calc_diff_2nd(b->fld.data(), fields.mp["w"]->fld.data(), fields.sd["evisc"]->fld.data(),
                                m.profs["bdiff"].data.data(), gd.dzhi.data(),
                                b->flux_bot.data(), b->flux_top.data(), diff.tPr, sloc,
                                mask_fieldh.fld.data(), stats.nmaskh.data());
        }
        else
            stats.calc_diff_2nd(b->fld.data(), m.profs["bdiff"].data.data(), gd.dzhi.data(),
                                get_buoyancy_diffusivity(), sloc, mask_fieldh.fld.data(), stats.nmaskh.data());
        fields.release_tmp(tmp);
    }
    // calculate the total fluxes
    stats.add_fluxes(m.profs["bflux"].data.data(), m.profs["bw"].data.data(), m.profs["bdiff"].data.data());

    // calculate the sorted buoyancy profile
    //stats->calc_sorted_prof(fields->sd["tmp1"]->data, fields->sd["tmp2"]->data, m->profs["bsort"].data);
    fields.release_tmp(b);


    if (bs_stats.swbasestate == Basestate_type::anelastic)
    {
        auto T = fields.get_tmp();
        get_thermo_field(*T, "T", false, true);
        stats.calc_mean(m.profs["T"].data.data(), T->fld.data(), no_offset, mask_field.fld.data(), stats.nmask.data());
        fields.release_tmp(T);
    }
}

template<typename TF>
void Thermo_dry<TF>::exec_dump(Dump<TF>& dump, unsigned long iotime)
{
    auto output = fields.get_tmp();

    for (auto& it : dumplist)
    {
        if (it == "b")
            get_thermo_field(*output, "b",false, true);
        else if (it == "T")
            get_thermo_field(*output, "T",false, true);
        else
        {
            master.print_error("Thermo dump of field \"%s\" not supported\n",it.c_str());
            throw std::runtime_error("Error in Thermo Dump");
        }
        dump.save_dump(output->fld.data(), it, iotime);
    }
    fields.release_tmp(output);
}

template<typename TF>
void Thermo_dry<TF>::exec_column(Column<TF>& column)
{
    const TF no_offset = 0.;
    auto output = fields.get_tmp();

    for (auto& it : dumplist)
    {
        if (it == "b")
            get_thermo_field(*output, "b",false, true);
        else if (it == "T")
            get_thermo_field(*output, "T",false, true);
        else
        {
            master.print_error("Thermo dump of field \"%s\" not supported\n",it.c_str());
            throw std::runtime_error("Error in Thermo Dump");
        }
        column.calc_column(it, output->fld.data(), no_offset);
    }
    fields.release_tmp(output);
}

template<typename TF>
void Thermo_dry<TF>::exec_cross(Cross<TF>& cross, unsigned long iotime)
{
    auto b = fields.get_tmp();

    if (swcross_b)
    {
        get_thermo_field(*b, "b", false, true);
        get_buoyancy_fluxbot(*b, true);
    }

    for (auto& it : crosslist)
    {
        if (it == "b")
            cross.cross_simple(b->fld.data(), "b", iotime);
        else if (it == "blngrad")
            cross.cross_lngrad(b->fld.data(), "blngrad", iotime);
        else if (it == "bbot")
            cross.cross_plane(b->fld_bot.data(), "bbot", iotime);
        else if (it == "bfluxbot")
            cross.cross_plane(b->flux_bot.data(), "bfluxbot", iotime);
        else if (it == "thlngrad")
            cross.cross_lngrad(fields.sp.at("th")->fld.data(), "thlngrad", iotime);
    }
    fields.release_tmp(b);
}

template class Thermo_dry<double>;
template class Thermo_dry<float>;
