#ifndef STATS_DNS
#define STATS_DNS

#include <netcdfcpp.h>
#include "grid.h"
#include "fields.h"
#include "mpiinterface.h"

class cstats_dns
{
  public:
    cstats_dns(cgrid *, cfields *, cmpi *);
    ~cstats_dns();

    int init();
    int create(int);
    int exec(int, double);

  private:
    cgrid   *grid;
    cfields *fields;
    cmpi    *mpi;

    bool allocated;
    bool initialized;

    NcFile *dataFile;
    NcDim  *z_dim, *zh_dim, *t_dim;
    NcVar  *z_var, *zh_var, *t_var, *iter_var;
    NcVar  *u_var , *v_var , *w_var , *s_var;
    NcVar  *evisc_var;
    NcVar  *u2_var, *v2_var, *w2_var, *s2_var;
    NcVar  *u3_var, *v3_var, *w3_var, *s3_var;
    NcVar  *ugrad_var, *vgrad_var, *sgrad_var;
    NcVar  *wu_var, *wv_var, *ws_var;
    NcVar  *udiff_var, *vdiff_var, *sdiff_var;
    NcVar  *uflux_var, *vflux_var, *sflux_var;
    NcVar  *u2_shear_var, *v2_shear_var, *tke_shear_var;
    NcVar  *u2_turb_var, *v2_turb_var, *w2_turb_var, *tke_turb_var;
    NcVar  *u2_visc_var, *v2_visc_var, *w2_visc_var, *tke_visc_var;
    NcVar  *u2_diss_var, *v2_diss_var, *w2_diss_var, *tke_diss_var;
    NcVar  *w2_pres_var, *tke_pres_var;
    NcVar  *u2_rdstr_var, *v2_rdstr_var, *w2_rdstr_var;
    NcVar  *w2_buoy_var, *tke_buoy_var;

    double *u , *v , *w , *s ;
    double *evisc;
    double *u2, *v2, *w2, *s2;
    double *u3, *v3, *w3, *s3;
    double *wu , *wv , *ws ;
    double *ugrad, *vgrad, *sgrad;
    double *udiff, *vdiff, *sdiff;
    double *uflux, *vflux, *sflux;
    double *u2_shear, *v2_shear, *tke_shear;
    double *u2_turb, *v2_turb, *w2_turb, *tke_turb;
    double *u2_visc, *v2_visc, *w2_visc, *tke_visc;
    double *u2_diss, *v2_diss, *w2_diss, *tke_diss;
    double *w2_pres, *tke_pres;
    double *u2_rdstr, *v2_rdstr, *w2_rdstr;
    double *w2_buoy, *tke_buoy;

    int calcmean     (double *, double *);
    int calcmoment   (double *, double *, double *, double, int);
    int calcflux     (double *, double *, double *, double *, int, int);
    int calcdiff     (double *, double *, double *, double);
    int calcgrad     (double *, double *, double *);
    int calctkebudget(double *, double *, double *, double *, double *,
                      double *, double *,
                      double *, double *,
                      double *, double *, double *,
                      double *, double *, double *, double *,
                      double *, double *, double *, double *,
                      double *, double *, double *, double *,
                      double *, double *,
                      double *, double *, double *,
                      double *, double *,
                      double *, double *, double);

    int nstats;
};
#endif
