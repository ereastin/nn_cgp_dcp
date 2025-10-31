import numpy as np
import xarray as xr

LEV = np.array([
    1000, 975, 950, 925, 900, 875, 850,
    825, 775, 700, 600, 550, 450, 400, 350, 300,
    250, 200, 150, 100, 70, 
    50, 40, 30, 20, 10, 7, 3
])

# -----------------------------------------------------------------------------
def regrid(ds, regrid_dict):
    if not regrid_dict['do']: return ds
    ds_grid = xr.open_dataset(regrid_dict['target_grid'])
    match regrid_dict['regrid_type']:
        case 'conservative':
            ds = ds.regrid.conservative(ds_grid)
        case 'nearest':
            ds = ds.regrid.nearest(ds_grid)
        case 'linear':
            ds = ds.regrid.linear(ds_grid)
        case 'cubic':
            ds = ds.regrid.cubic(ds_grid)
        case _:
            print('undefined regridding type')
            sys.exit()
    return ds

# -----------------------------------------------------------------------------
def _perturb(ds, instructions):
    perturb_var = instructions['var']
    perturb_type = instructions['type']
    scale = instructions['scale']
    region = instructions['region']
    invert = instructions['invert']
    levels = instructions['levels']
 
    # select for specific region(s)
    if 'plev' in ds.coords.keys(): ds = ds.rename({'plev': 'lev'})
    # need the values call?
    plvls = ds.coords['lev'].values[()]
    lats, lons = ds.coords['lat'].values[()], ds.coords['lon'].values[()]
    if region is not None:
        region = region.expand_dims(lev=plvls)
    else:
        region = xr.ones_like(ds[perturb_var])
    if invert: region = xr.where(region > 0, np.nan, 1)

    if levels is not None:
        m = np.isin(LEV, levels)
        sel_levels = LEV[~m]
        region.loc[dict(lev=sel_levels)] = np.nan

    # do perturbation
    match perturb_type:
        case 'scale':  # can we do a per-level scaling? can then move mean 4X vprofile to CTRL
            ds[perturb_var].loc[dict(lev=plvls)] = xr.where(region > 0, ds[perturb_var] * scale, ds[perturb_var])
        case 'vshear':  # low-level vertical wind shear thought to be important for MCS (Rotunno '88), also LLJs for moisture?
            # remove per-column vertical shear -> set field at each cell to mean of cell across p-lvls
            # TODO: this is way slow with the compute.. but can't multi-dim index with dask.. ?/
            ds = ds.compute()
            masked = xr.where(region > 0, ds[perturb_var], region).mean(dim='lev', skipna=True)
            ds[perturb_var].loc[dict(lat=lats, lon=lons)] = xr.where(region > 0, masked, ds[perturb_var])
        case 'hshear':
            # remove horizontal shear -> set field at each p-lvl to mean of p-lvl
            masked = xr.where(region > 0, ds[perturb_var], region).mean(dim=['lat', 'lon'], skipna=True)
            ds[perturb_var].loc[dict(lev=plvls)] = xr.where(region > 0, masked, ds[perturb_var])
        case 'mean':
            ds[perturb_var].loc[dict(lev=plvls)] = ds[perturb_var].mean(dim='time')
        case 'shuffle':
            #mask = region > 0
            #mask = mask.compute()
            cp_da = ds[perturb_var].copy()
            for t in cp_da.time.values:
                for p in [levels]:
                    vals = np.asarray(cp_da.sel(time=t, lev=p).values)
                    s = vals.shape
                    a = vals.flatten()
                    #m = mask.sel(time=t, lev=p)
                    # each (81, 145) 'images'
                    #v = vals.where(m).values.flatten()
                    #v = v[~np.isnan(v)]
                    np.random.shuffle(a)
                    #a[m.values] = v
                    cp_da.loc[dict(time=t, lev=p)] = a.reshape(s)

            ds[perturb_var] = cp_da
    return ds

# -----------------------------------------------------------------------------
def perturb(ds, perturb_dict):
    if perturb_dict == {}: return ds
    for _id, instr in perturb_dict.items():
        ds = _perturb(ds, instr)
    return ds

# ---------------------------------------------------------------------------------
def build_exp_note(perturb_dict):
    if perturb_dict == {}:
        return ''
    else:        
        # scale, vshear, hshear, shuffle, mask?, levels?
        note = ''
        for _id, instr in perturb_dict.items():
            v, t, s = instr['var'], instr['type'], instr['scale']
            if type(s) != float or type(s) != int:
                s = 'vprof'
            ss = f'{t}{s}' if t == 'scale' else f'{t}'
            r, i, l = instr['region'], instr['invert'], instr['levels']
            if r is not None:
                if i:
                    rr = '-IM'
                else:
                    rr = '-M'
            else:
                rr = ''
            if l is not None:
                if type(l) == np.int64:
                    ll = f'-L{l}'
                else:
                    ll = f'-L{str(l[0])}-{str(l[-1])}'  # this ok?
            else:
                ll = ''
            note += f'_{_id}-{v}-{ss}{rr}{ll}'
        return note

# ---------------------------------------------------------------------------------
def lat_wtd_sum(da):
    wt = np.cos(np.radians(da.lat))
    return da.weighted(wt).sum(dim=['lat', 'lon'], skipna=True)

# ---------------------------------------------------------------------------------
def lat_wtd_mean(da):
    wt = np.cos(np.radians(da.lat))
    return da.weighted(wt).mean(dim=['lat', 'lon'], skipna=True)

# ---------------------------------------------------------------------------------
def latonly_wtd_mean(da):
    wt = np.cos(np.radians(da.lat))
    return da.weighted(wt).mean(dim='lat', skipna=True)

# ---------------------------------------------------------------------------------
def calc_qsat(T, p):
    # modified Tetens formula; see Simmons et al. (1999)
    # 'Stratospheric water vapour and tropical tropopause temperatures in ECMWF analyses and multi-year simulations'
    eps = 0.622  # ratio of dry air and water vapor gas constants

    def esat_ice(T):
        a1 = 611.21
        a3 = 22.587
        a4 = -0.7
        T0 = 273.16  # K
        return a1 * np.exp(a3 * (T - T0) / (T - a4))

    def esat_liq(T):
        a1 = 611.21
        a3 = 17.502
        a4 = 32.19
        T0 = 273.16  # K
        return a1 * np.exp(a3 * (T - T0) / (T - a4))

    def esat_mix(T):
        T0 = 273.16  # K
        Tice = T0 - 23  # K
        return esat_ice(T) + (esat_liq(T) - esat_ice(T)) * ((T - Tice) / (T0 - Tice)) ** 2

    def esat(T):
        T0 = 273.16
        Tice = T0 - 23
        liq = xr.where(T > T0, esat_liq(T), 0)
        ice = xr.where(T < Tice, esat_ice(T), 0)
        mix = xr.where((T <= T0) & (T >= Tice), esat_mix(T), 0)
        return liq + ice + mix

    return eps * esat(T) / (p - (1 - eps) * esat(T))

# ---------------------------------------------------------------------------------
def calc_virtual_temp(T, q):
    return T * (1 + 0.61 * q)

