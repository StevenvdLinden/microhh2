/*
 * MicroHH
 * Copyright (c) 2011-2013 Chiel van Heerwaarden
 * Copyright (c) 2011-2013 Thijs Heus
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

#ifndef STATS
#define STATS

#include <netcdfcpp.h>

// forward declarations to reduce compilation time
class cmaster;
class cmodel;
class cgrid;
class cfields;

// struct for profiles
struct profvar
{
  NcVar *ncvar;
  double *data;
};

// struct for time series
struct tseriesvar
{
  NcVar *ncvar;
  double data;
};

// typedefs for containers of profiles and time series
typedef std::map<std::string, profvar> profmap;
typedef std::map<std::string, tseriesvar> tseriesmap;

// structure
struct filter
{
  std::string name;
  NcFile *dataFile;
  NcDim  *z_dim, *zh_dim, *t_dim;
  NcVar  *t_var, *iter_var;
  profmap profs;
  tseriesmap tseries;
};

typedef std::map<std::string, filter> filtermap;

class cstats
{
  public:
    cstats(cmodel *);
    ~cstats();

    int readinifile(cinput *);
    int init(double);
    int create(int);
    unsigned long gettimelim(unsigned long);
    int getfilter(cfield3d *, filter *);
    int exec(int, double, unsigned long);
    int dostats();
    std::string getsw();

    // container for all stats, filter as uppermost in hierarchy
    filtermap filters;
    int *filtercount;

    // interface functions
    // profmap profs;
    // tseriesmap tseries;

    int addprof(std::string, std::string, std::string, std::string);
    int addfixedprof(std::string, std::string, std::string, std::string, double *);
    int addtseries(std::string, std::string, std::string);

    int calcmean    (double *, double *, double);
    int calcmean    (double *, double *, double, const int[3], double *, int *);
    // int calcmoment  (double *, double *, double *, double, int);
    int calcmoment  (double *, double *, double *, double, const int[3], double *, int *);
    int calcdiff_2nd(double *, double *, double *, double *, double *, double *, double);
    int calcdiff_4th(double *, double *, double *, double, const int[3], double *, int *);
    // int calcgrad_2nd(double *, double *, double *);
    int calcgrad_2nd(double *, double *, double *, const int[3], double *, int *);
    int calcgrad_4th(double *, double *, double *, const int[3], double *, int *);
    // int calcflux_2nd(double *, double *, double *, double *, int, int);
    int calcflux_2nd(double *, double *, double *, double *, double *, double *, const int[3], double *, int *);
    int calcflux_4th(double *, double *, double *, double *, int, int);
    int addfluxes   (double *, double *, double *);
    // int calccount   (double *, double *, double);
    int calccount   (double *, double *, double, double *, int *);
    int calcpath    (double *, double *);
    int calccover   (double *, double *, double);

  private:
    // NcFile *dataFile;
    // NcDim  *z_dim, *zh_dim, *t_dim;
    // NcVar  *t_var, *iter_var;

    double *umodel, *vmodel;

    int nstats;

    // filters
    int calcfilter(double *, double *, double *, int *);

  protected:
    cmodel  *model;
    cgrid   *grid;
    cfields *fields;
    cmaster *master;

    double sampletime;
    unsigned long isampletime;

    std::string swstats;
};
#endif
