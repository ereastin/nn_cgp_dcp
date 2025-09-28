import xarray as xr
from dask.distributed import Client, LocalCluster
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append('/home/eastinev/ai')
import file_utils as futils
import utils
import paths as pth
from metrics import *
from functools import partial

# ---------------------------------------------------------------------------------
def main():
    n_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    cluster = LocalCluster(n_workers=n_cpus)
    print(cluster)
    with Client(cluster) as client:
        print(client)
        model_name = 'mcs_JJA_F03h'
        season = model_name.split('_')[1]

        ## LSF stuff
        def _select_batch(ds, **kwargs):
            return ds.sel(**kwargs)

        def open_mf(filepath, drop_vars=[], **kwargs):
            preproc_fn = partial(_select_batch, **kwargs) 
            ds = xr.open_mfdataset(
                filepath,
                preprocess=preproc_fn,
                drop_variables=drop_vars,
                concat_dim='time',
                data_vars='minimal',
                coords='minimal',
                combine='nested',
                compat='override',
                join='override',
                parallel=True,
                chunks='auto',
                engine='h5netcdf'
            )
            return ds

        """
        filepath = os.path.join(pth.SCRATCH, 'cus_cesm', 'P*.CTRL.nc')
        pr = open_mf(filepath)
        Rx1day = pr.resample(time='D').sum(dim='time').dropna('time', how='all')
        #Rx1day_both = Rx1day.groupby('time.year').max(dim='time').mean(dim='year')
        #Rx1day_spr = Rx1day.sel(time=Rx1day.time.dt.month.isin([3, 4, 5]))
        #Rx1day_sum = Rx1day.sel(time=Rx1day.time.dt.month.isin([6, 7, 8]))
        #Rx1day_spr = Rx1day_spr.groupby('time.year').max(dim='time').mean(dim='year')
        #Rx1day_sum = Rx1day_sum.groupby('time.year').max(dim='time').mean(dim='year')
        # believe this is datetime at which max daily precip occurs
        max_day = Rx1day.groupby('time.year').apply(lambda da: da.idxmax(dim='time'))

        lsf_path = os.path.join(pth.SCRATCH, 'cus_cesm', 'LSF*.CTRL.nc')
        lsf = open_mf(lsf_path, drop_vars=['U', 'V', 'Z3'])  # keep T, OMEGA, Q
        # new Dataset() for storing scaling results
        qs_da = xr.full_like(max_day, 0).expand_dims(dim={'plev': lsf.plev}).rename({'PRECT': 'QS'})
        omega_da = xr.full_like(max_day, 0).expand_dims(dim={'plev': lsf.plev}).rename({'PRECT': 'OMEGA'})
        rho_da = xr.full_like(max_day, 0).expand_dims(dim={'plev': lsf.plev}).rename({'PRECT': 'RHO'})
        T_da = xr.full_like(max_day, 0).expand_dims(dim={'plev': lsf.plev}).rename({'PRECT': 'T'})
        scaling = xr.merge([qs_da, omega_da, rho_da, T_da])
        lsf_daily = lsf.resample(time='D').mean(dim='time').dropna('time', how='all')

        # gotta be an easier way than this right? ... this will take fuckn forever..
        for yr in max_day.year:
            for lat in max_day.lat:
                for lon in max_day.lon:
                    dt = max_day.sel(year=yr, lat=lat, lon=lon)['PRECT'].values
                    gridbox = lsf_daily.sel(time=dt, lat=lat, lon=lon)
                    Tv = utils.calc_virtual_temp(gridbox['T'], gridbox['Q'])
                    Rd = 287  # J / kg * K
                    scaling['RHO'].loc[dict(lat=lat, lon=lon, year=yr)] = gridbox['plev'] / (Rd * Tv)  # density, for mass-weighted integral
                    scaling['QS'].loc[dict(lat=lat, lon=lon, year=yr)] = utils.calc_qsat(gridbox['T'], gridbox['plev'])
                    scaling['OMEGA'].loc[dict(lat=lat, lon=lon, year=yr)] = gridbox['OMEGA']
                    scaling['T'].loc[dict(lat=lat, lon=lon, year=yr)] = gridbox['T']
                    print(scaling.sel(year=yr, lat=lat, lon=lon).values)
                    exit()
        """

        # TODO: create 'animations' for a few events showing evolution of q, omega, and P in panels.?

        ## Comparing MERRA and MSWEP precip for 2004-2020
        cesm_filepath = os.path.join(pth.SCRATCH, 'cus_cesm', 'LSF*.CTRL.nc')
        cesm_lsf = open_mf(cesm_filepath, drop_vars=['T', 'U', 'V', 'Z3'], plev=[925, 550])
        cesm_p = xr.open_dataset('./CESM_sum_CESM.CTRL.nc')
        dl_p = xr.open_dataset('./DL_sum_CESM.CTRL.nc')
        #mswep = open_mf()
        #mswep = mswep.isel(time=mask)
        #mswep = mswep['precipitation'].rename('precip')
        #merra = xr.open_dataset('./MERRA_precip.nc')
        #merra = merra['PRECTOT'].rename('precip')
        #mask = ~(dl.time.dt.month == 9)
        #complete_spr = [mswep.sel(time=mswep.time.dt.month.isin([3, 4, 5])), merra.sel(time=merra.time.dt.month.isin([3, 4, 5]))]
        #complete_sum = [mswep.sel(time=mswep.time.dt.month.isin([6, 7, 8])), merra.sel(time=merra.time.dt.month.isin([6, 7, 8]))]

        ## CESM testing precip stuff 1979-1983/4
        #dl = xr.open_dataset(f'./DL_{season}_CESM.CTRL.nc')
        #cesm = xr.open_dataset(f'./CESM_{season}_CESM.CTRL.nc')
        #mswep = xr.open_dataset(f'./MSWEP_{season}_CESM.CTRL.nc')
        #dl_4x = xr.open_dataset(f'./DL_{season}_CESM.2K.nc')
        #cesm_4x = xr.open_dataset(f'./CESM_{season}_CESM.2K.nc')
        #complete_4x = [dl_4x, cesm_4x]
        #complete_ctrl = [dl, cesm, mswep]
        #complete = complete_ctrl + complete_4x

        # CUS 3D Inc MERRA2 grid
        gridshape = (53, 65)  # (H, W)
        extent = (-110, -70, 25, 51)
        lons, lats = np.linspace(extent[0], extent[1], gridshape[1]), np.linspace(extent[2], extent[3], gridshape[0])

        # plotting directives
        precip_plot_params = {'extent': extent, 'lons': lons, 'lats': lats, 'cmap': 'pprecip'}
        bias_plot_params = {'extent': extent, 'lons': lons, 'lats': lats, 'cmap': 'bias'}
        clima_plot_params = {'extent': extent, 'lons': lons, 'lats': lats, 'cmap': 'clima'}

        utils.plot_mean(
            {'Mean of Annual Max Single Day Precip': Rx1day_spr['PRECT']},
            clima_plot_params,
            model_name,
            note=note + '_Rx1day_spr',
            bias=False
        )
        utils.plot_mean(
            {'Mean of Annual Max Single Day Precip': Rx1day_sum['PRECT']},
            clima_plot_params,
            model_name,
            note=note + '_Rx1day_sum',
            bias=False
        )
        exit()
        #get_pdf(complete)
        #precip(complete, season, precip_plot_params, model_name, note)
        #seasonal_stats(complete, bias_plot_params, clima_plot_params, model_name, note)
        #annual_stats(complete, bias_plot_params, clima_plot_params, model_name, note)
        #daily_stats(complete)
        #hourly_stats(complete, bias_plot_params, clima_plot_params, season, model_name, note)

# ---------------------------------------------------------------------------------
def get_pdf(complete):
    # Plot PDF of rainfall
    # TODO: still not super sure of this one
    daily = [da.resample(time='D').sum(dim='time').dropna('time', how='all') for da in complete]
    out = [da.where(da > 0.3).to_dataarray().data.flatten() for da in daily]
    mask = [~np.isnan(arr) for arr in out]
    out = [arr[m] for arr, m in zip(out, mask)]

    colors = ['blue', 'red', 'black', 'green', 'gold']
    styles = ['solid', 'solid', 'solid', 'solid', 'solid']
    labels = ['DL', 'CESM', 'MSWEP', 'DL4x', 'CESM4x']
    ax = sns.histplot(data=out, fill=False, element='poly', stat='probability', log_scale=True)
    ax.set(ylabel='Probability', xlabel='Daily Accumulated Precipitation [mm/day]')
    plt.legend(labels=labels)
    plt.savefig('./test.png', dpi=300)

# ---------------------------------------------------------------------------------
def precip(complete, season, plot_params, model_name, note):
    # Known Sumatra Squall events (not in test apparently)
    # selected = [(2016, 7, 12), (2014, 6, 11), (2014, 6, 12)]
    match season:
        case 'both':
            selected = ['2017-08-04', '2008-05-12', '2007-04-18', '2015-07-26']
        case 'sum':
            #selected = ['2008-07-23', '2010-06-30', '2010-08-24', '2009-07-21']
            # dry days
            #selected += ['2010-06-25', '2015-07-10', '2019-07-31', '2005-07-12']
            selected = ['1983-06-25', '1980-06-19', '1983-06-19']  # highest total rain
            #selected = ['1983-06-27', '1980-07-21', '1979-06-20']
        case 'spr':
            #selected = ['2010-03-17', '2009-05-10', '2016-04-10', '2013-04-25']
            # dry days
            #selected += ['2010-03-05', '2018-04-18', '2007-04-12', '2013-03-02']
            #selected = ['1982-03-01', '1979-04-16', '1981-04-06']  # highest toal rain
            selected = ['1979-04-17', '1980-03-09', '1982-04-10']

    complete_pred, complete_obs = complete[0], complete[1]
    for dt in selected:
        # will select all present times of specified day
        pred_sel = complete_pred.sel(time=dt)['precip'].as_numpy()
        obs_sel = complete_obs.sel(time=dt)['precip'].as_numpy()
        utils.plot_precip(
            pred_sel,
            obs_sel,
            dt,
            plot_params,
            model_name,
            note=note
        )

# ---------------------------------------------------------------------------------
def annual_stats(complete, bias_plot_params, clima_plot_params, model_name, note):
    # per-year daily mean
    annual = [da.resample(time='D').sum(dim='time').dropna('time', how='all').groupby('time.year').mean(dim='time') for da in complete]

    # total annual mean daily precip
    labels = ['DL', 'CESM']
    climatology = {}
    for da, _id in zip(annual, labels):
        climatology |= {str(yr) + _id: da.sel(year=yr)['precip'].as_numpy() for yr in da.year.values}

    utils.plot_mean(
        climatology,
        clima_plot_params,
        model_name,
        note=note + '_annual',
        bias=False
    )

    # total annual mean daily precip bias
    bias = annual[0] - annual[1]
    utils.plot_mean(
        {str(yr): bias.sel(year=yr)['precip'].as_numpy() for yr in bias.year.values},
        bias_plot_params,
        model_name,
        note=note + '_annual',
        bias=True
    )

# ---------------------------------------------------------------------------------
def daily_stats(complete):
    # resample to regional latitude-weighted daily means
    daily = [_lat_wtd_mean(da.resample(time='D').mean(dim='time').dropna('time', how='all')) for da in complete]
    diff = (daily[0] - daily[1]).sortby(lambda x: x, ascending=False)
    print(diff.values[:8])
    print(diff.time.values[:8])
    return
    min_obs = [da.sortby(lambda x: x, ascending=False) for da in daily]
    labels = ['DL', 'CESM']
    for item, _id in zip(min_obs, labels):
        print(_id)
        print(item.values[:8])
        print(item.time.values[:8])

# ---------------------------------------------------------------------------------
def seasonal_stats(complete, bias_plot_params, clima_plot_params, model_name, note):
    seasonal = [da.resample(time='D').sum(dim='time').dropna('time', how='all').groupby('time.season').mean(dim='time') for da in complete]

    # total seasonal mean daily precip
    labels = ['MSWEP', 'MERRA']
    climatology = {}
    for da, _id in zip(seasonal, labels):
        climatology |= {ssn + _id: da.sel(season=ssn).as_numpy() for ssn in da.season.values}

    utils.plot_mean(
        climatology,
        clima_plot_params,
        model_name,
        note=note + '_seasonal',
        bias=False
    )

    # total seasonal mean daily precip bias
    #ctrl = xr.open_dataset('./pred_sum_ctrl.nc')
    bias = seasonal[1] - seasonal[0]
    utils.plot_mean(
        {ssn: bias.sel(season=ssn).as_numpy() for ssn in bias.season.values},
        bias_plot_params,
        model_name,
        note=note + '_seasonal',
        bias=True
    )

# ---------------------------------------------------------------------------------
def hourly_stats(complete, bias_plot_params, clima_plot_params, season, model_name, note):
    # sub-region slices
    NGP_slat, SGP_slat, slat, slon = slice(40, 48), slice(31, 40), slice(31, 48), slice(-102, -85)

    # TODO: this the correct order of things? want for just the whole warm-season?
    hourly = [da.groupby('time.hour').mean(dim='time') for da in complete]
    NGP = [_lat_wtd_mean(da.sel(lat=NGP_slat, lon=slon))['precip'] for da in hourly]  # order as pred, obs, (comp)
    SGP = [_lat_wtd_mean(da.sel(lat=SGP_slat, lon=slon))['precip'] for da in hourly]

    fig, [ax0, ax1] = plt.subplots(1, 2, sharey=True)
    colors = ['blue', 'red', 'black', 'green', 'gold']
    styles = ['solid', 'solid', 'solid', 'solid', 'solid']
    labels = ['DL', 'CESM', 'MSWEP', 'DL4x', 'CESM4x']
    for da, _id, c in zip(NGP, labels, colors): da.plot.line(ax=ax0, color=c, label=_id)
    for da, _id, c in zip(SGP, labels, colors): da.plot.line(ax=ax1, color=c, label=_id)
    ax0.set(title='NGP', xlabel='UTC Hour', ylabel='mean regional accum. [mm]', xlim=(0, 21))
    ax1.set(title='SGP', xlabel='UTC Hour', ylabel='', xlim=(0, 21))
    ax0.legend(frameon=False, fancybox=False, fontsize='small')
    fig.suptitle(f'Regional Mean Diurnal Cycle for {season}')
    fig.tight_layout()
    plt.savefig(os.path.join(pth.MODEL_OUT, f'{model_name}', f'line_diurnal_{note}.png'), dpi=300)
    return

    # TODO: Hovmoller diagrams.? would need plot to do interpolation or it's ugly

    # Spatially-explicit regional plot of diurnal cycle
    utils.plot_mean(
        {'PRED ' + str((hr - 6) % 24) + 'CST': hourly_pred.sel(hour=hr)['precip'].as_numpy() for hr in hours} | {'OBS ' + str((hr - 6) % 24) + 'CST': hourly_obs.sel(hour=hr)['precip'].as_numpy() for hr in hours},
        clima_plot_params,
        model_name,
        note=note + '_diurnal',
        bias=False
    )
    utils.plot_mean(
        {'BIAS ' + str((hr - 6) % 24) + 'CST': bias.sel(hour=hr)['precip'].as_numpy() for hr in hours},
        bias_plot_params,
        model_name,
        note=note + '_diurnal_bias',
        bias=True
    )
    return

# ---------------------------------------------------------------------------------
def _lat_wtd_sum(da):
    wt = np.cos(np.radians(da.lat))
    return da.weighted(wt).sum(dim=['lat', 'lon'])

# ---------------------------------------------------------------------------------
def _lat_wtd_mean(da):
    wt = np.cos(np.radians(da.lat))
    return da.weighted(wt).mean(dim=['lat', 'lon'])

# ---------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
