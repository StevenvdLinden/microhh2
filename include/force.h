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

#ifndef FORCE_H
#define FORCE_H

#include <vector>
#include <string>
#include <map>

class Master;
class Input;
template<typename> class Timeloop;
template<typename> class Grid;
template<typename> class Fields;
template<typename> class Field3d_operators;

/**
 * Class for the right-hand side terms that contain large-scale forcings
 * This class contains the large-scale pressure forcings, either in flux for or through a
 * geostrophic wind and a coriolis force. Furthermore, a large scale vertical velocity can
 * be imposed that advects the scalars through the domain. Profiles of sources/sinks can be
 * assigned to all scalars.
 */

enum class Large_scale_pressure_type {disabled, fixed_flux, geo_wind};
enum class Large_scale_tendency_type {disabled, enabled};
enum class Large_scale_subsidence_type {disabled, enabled};
enum class Nudging_type {disabled, enabled};

template<typename TF>
class Force
{
    public:
        Force(Master&, Grid<TF>&, Fields<TF>&, Input&); ///< Constructor of the force class.
        ~Force();                                       ///< Destructor of the force class.

        void init();           ///< Initialize the arrays that contain the profiles.
        void create(Input&, Data_block&);   ///< Read the profiles of the forces from the input.
        void exec(double);     ///< Add the tendencies belonging to the large-scale processes.

        void update_time_dependent(Timeloop<TF>&); ///< Update the time dependent parameters.

        std::vector<std::string> lslist;        ///< List of variables that have large-scale forcings.
        std::map<std::string, std::vector<TF>> lsprofs; ///< Map of profiles with forcings stored by its name.

        std::vector<std::string> nudgelist;        ///< List of variables that are nudged to a provided profile
        std::map<std::string, std::vector<TF>> nudgeprofs; ///< Map of nudge profiles stored by its name.

        // GPU functions and variables
        void prepare_device();
        void clear_device();

        std::map<std::string, TF*> lsprofs_g;    ///< Map of profiles with forcings stored by its name.
        std::map<std::string, TF*> nudgeprofs_g; ///< Map of nudging profiles stored by its name.

        // Accessor functions
        //std::string get_switch_lspres()      { return swlspres; }
        //TF      get_coriolis_parameter() { return fc;       }


    private:
        Master& master;
        Grid<TF>& grid;
        Fields<TF>& fields;
        Field3d_operators<TF> field3d_operators;

        // Internal switches for various forcings
        Large_scale_pressure_type swlspres;
        Large_scale_tendency_type swls;
        Large_scale_subsidence_type swwls;
        Nudging_type swnudge;

        TF uflux; ///< Mean velocity used to enforce constant flux.
        TF fc;    ///< Coriolis parameter.

        std::vector<TF> ug;  ///< Pointer to array u-component geostrophic wind.
        std::vector<TF> vg;  ///< Pointer to array v-component geostrophic wind.
        std::vector<TF> wls; ///< Pointer to array large-scale vertical velocity.

        std::vector<TF> nudge_factor;  ///< Height varying nudging factor (1/s)

        struct Time_dep
        {
            Time_dep() : sw(false) {}; // Initialize the switches to zero.
            Time_dep(const Time_dep&) = delete; // Delete the copy assignment to prevent mistakes.
            Time_dep& operator=(const Time_dep&) = delete; // Delete the copy assignment to prevent mistakes.
            Time_dep(Time_dep&&) = delete; // Delete the move assignment to prevent mistakes.
            Time_dep& operator=(Time_dep&&) = delete; // Delete the move assignment to prevent mistakes.
            
            bool sw;
            std::vector<std::string> vars;
            std::map<std::string, std::vector<double>> time;
            std::map<std::string, std::vector<TF>> data;
            std::map<std::string, TF*> data_g;
        };

        Time_dep tdep_ls;
        Time_dep tdep_geo;
        Time_dep tdep_wls;
        Time_dep tdep_nudge;

        void create_timedep(Time_dep&, std::string);
        void update_time_dependent_profs(Timeloop<TF>&, std::map<std::string, std::vector<TF>>, Time_dep&);
        void update_time_dependent_prof(Timeloop<TF>&, std::vector<TF>, Time_dep&, const std::string&);

        // GPU functions and variables
        TF* ug_g;  ///< Pointer to GPU array u-component geostrophic wind.
        TF* vg_g;  ///< Pointer to GPU array v-component geostrophic wind.
        TF* wls_g; ///< Pointer to GPU array large-scale vertical velocity.
        TF* nudge_factor_g; ///< Pointer to GPU array nudge factor.

        #ifdef USECUDA
        void update_time_dependent_profs_g(Timeloop<TF>&, std::map<std::string, TF*>, Time_dep&);
        void update_time_dependent_prof_g(Timeloop<TF>&, TF*, Time_dep&, const std::string&);
        #endif
};
#endif
