import xarray as xr
from dask.distributed import Client, LocalCluster
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import seaborn as sns
import os
from argparse import ArgumentParser
import sys
sys.path.append('/home/eastinev/ai')
import file_utils as futils
import utils
import paths as pth
from metrics import *
from functools import partial
import cftime
import pandas as pd

# Lin et al. uses 31-52N, -98--89E
NGP_SLAT, SGP_SLAT, SLAT, SLON = slice(40, 48), slice(31, 40), slice(31, 48), slice(-105, -85)#slice(-102, -85)
GP_SLAT, GP_SLON = slice(35, 45), slice(-105, -85)
CUS_LON = slice(-110, -70)
CUS_LAT = slice(25, 51)
R_THRESH = 0  # mm /day

# CUS 3D Inc MERRA2 grid
GRIDSHAPE = (53, 65)  # (H, W)
EXTENT = (-110, -70, 25, 51)
LONS = np.linspace(EXTENT[0], EXTENT[1], GRIDSHAPE[1])
LATS = np.linspace(EXTENT[2], EXTENT[3], GRIDSHAPE[0])
PLOT_PARAMS = {'extent': EXTENT, 'lons': LONS, 'lats': LATS}

## ================================================================================
def main():
    parser = ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('-e', '--exp', type=str, default='')
    args = parser.parse_args()
    model_name = args.model_name
    cesm_exp = args.exp

    print(f'Running analysis on {model_name}')
    n_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    cluster = LocalCluster(n_workers=n_cpus)
    print(cluster)
    with Client(cluster) as client:
        print(client)
        season = model_name.split('_')[1]

        ## O'Gorman scaling stuff..
        """
        pr = xr.open_dataset('./CESM.CTRL_1979-1983.nc')
        Rx1day = pr.resample(time='D').sum(dim='time').dropna('time', how='all')
        #Rx1day_both = Rx1day.groupby('time.year').max(dim='time').mean(dim='year')
        #Rx1day_spr = Rx1day.sel(time=Rx1day.time.dt.month.isin([3, 4, 5]))
        Rx1day = Rx1day.sel(time=Rx1day.time.dt.month.isin([6, 7, 8]))
        #Rx1day_spr = Rx1day_spr.groupby('time.year').max(dim='time').mean(dim='year')
        #Rx1day = Rx1day_sum.groupby('time.year').max(dim='time').mean(dim='year')
        # believe this is datetime at which max daily precip occurs
        max_day = Rx1day.groupby('time.year').apply(lambda da: da.idxmax(dim='time'))
        ## compare with DL how? ig do the same thing w DL days (if different?)
        #dl_pr = xr.open_dataset(f'./models/{model_name}/MODEL_PRED_CESM.CTRL_ctrl.nc')
        #dl_Rx1day = dl_pr.resample(time='D').sum(dim='time').dropna('time', how='all')
        #dl_max_day = dl_Rx1day.groupby('time.year').apply(lambda da: da.idxmax(dim='time'))

        lsf_path = os.path.join(pth.SCRATCH, 'cus_cesm', 'LSF*.CTRL.nc')
        lsf = open_mf(lsf_path, drop_vars=['U', 'V', 'Z3'])  # keep T, OMEGA, Q
        lsf_daily = lsf.resample(time='D').mean(dim='time').dropna('time', how='all')
        # new Dataset() for storing scaling results
        qs_da = xr.full_like(max_day, 0).expand_dims(dim={'plev': lsf.plev}).rename({'PRECT': 'QS'})
        omega_da = xr.full_like(max_day, 0).expand_dims(dim={'plev': lsf.plev}).rename({'PRECT': 'OMEGA'})
        rho_da = xr.full_like(max_day, 0).expand_dims(dim={'plev': lsf.plev}).rename({'PRECT': 'RHO'})
        T_da = xr.full_like(max_day, 0).expand_dims(dim={'plev': lsf.plev}).rename({'PRECT': 'T'})
        alll = xr.merge([qs_da, omega_da, rho_da, T_da])
        scaling = alll.copy(deep=True)

        # gotta be an easier way than this right? ... this will take fuckn forever..
        print('hello')
        for yr in max_day.year:
            for lat in max_day.lat:
                print(lat)
                for lon in max_day.lon:
                    dt = max_day.sel(year=yr, lat=lat, lon=lon)['PRECT'].values
                    gridbox = lsf_daily.sel(time=dt, lat=lat, lon=lon)
                    Tv = utils.calc_virtual_temp(gridbox['T'], gridbox['Q'])
                    Rd = 287  # J / kg * K
                    scaling['RHO'].loc[dict(lat=lat, lon=lon, year=yr)] = gridbox['plev'] * 100 / (Rd * Tv)  # density, for mass-weighted integral
                    scaling['QS'].loc[dict(lat=lat, lon=lon, year=yr)] = utils.calc_qsat(gridbox['T'], gridbox['plev'])
                    scaling['OMEGA'].loc[dict(lat=lat, lon=lon, year=yr)] = gridbox['OMEGA']
                    scaling['T'].loc[dict(lat=lat, lon=lon, year=yr)] = gridbox['T']

        scaling.to_netcdf('./cesm_scaling.nc', engine='netcdf4')
        return
        """

        ## full analysis
        if cesm_exp != '':
            dl_p_cesm = xr.open_dataset(f'./models/{model_name}/MODEL_PRED_CESM.{cesm_exp}.nc')
            times = dl_p_cesm.time  # already as cftime objs
            cesm_p = xr.open_dataset(f'./CESM.{cesm_exp}_1979-1983.nc').sel(time=times).rename({'PRECT': 'precip'})
            mswep_p_cesm = xr.open_dataset(f'./MSWEP_1979-1983.nc').convert_calendar('noleap', use_cftime=True).sel(time=times)
            complete = [dl_p_cesm, mswep_p_cesm, cesm_p]
            note = f'CESM.{cesm_exp}'
            cesm = True
        else:
            dl_p_test = xr.open_dataset(f'./models/{model_name}/MODEL_PRED.nc')
            mswep_p_test = xr.open_dataset(f'./models/{model_name}/MSWEP.nc')
            complete = [dl_p_test, mswep_p_test]
            note = 'ctrl'
            cesm = False

        get_pdf(complete, model_name, note)
        #precip(complete, PLOT_PARAMS, model_name, note)
        seasonal_stats(complete, PLOT_PARAMS, model_name, note)
        #annual_stats(complete, PLOT_PARAMS, model_name, note)
        #daily_stats(complete)
        hourly_stats(complete, PLOT_PARAMS, model_name, note)
        return
        t = np.datetime64(weekly_stats(complete)) - np.timedelta64(1, 'D')
        tend = t + np.timedelta64(7, 'D')
        time = slice(t.astype(str).split('T')[0], tend.astype(str).split('T')[0])
        print(time)
        animate(model_name, PLOT_PARAMS, time, cesm=cesm)
        #return

# ---------------------------------------------------------------------------------
def animate(model_name, plot_params, time, cesm=True):
    tsel = time.start[:7].replace('-', '_')
    kwargs = {'lat': CUS_LAT, 'lon': CUS_LON, 'time': time}
    pr_kwargs = {'drop_vars': [], 'time': time}
    if cesm:
        lsf_fpath = os.path.join(pth.SCRATCH, 'cus_cesm', f'LSF_{tsel}*.CTRL.nc')
        obs_p_fpath = os.path.join('CESM.CTRL_1979-1983.nc'),
        dl_p_fpath = os.path.join('models', model_name, 'MODEL_PRED_CESM.CTRL.nc')
        kwargs |= {'drop_vars': ['T', 'U', 'V', 'Z3']}
    else:
        lsf_fpath = os.path.join(pth.SCRATCH, 'cus', f'LSF_{tsel}*.nc')
        kwargs |= {'drop_vars': ['T', 'U', 'V', 'H']}
        obs_p_fpath = os.path.join('models', model_name, f'MSWEP.nc')
        dl_p_fpath = os.path.join('models', model_name, 'MODEL_PRED.nc')

    lsf = open_mf(
        lsf_fpath,
        **kwargs
    )
    # TODO: need to fix the datetime stuff i think.?
    if cesm: lsf = lsf.rename({'plev': 'lev', 'Q': 'QV'})
    obs_p = open_mf(
        obs_p_fpath,
        **pr_kwargs
    )
    if cesm: obs_p = obs_p.rename({'PRECT': 'precip'})
    dl_p = open_mf(
        dl_p_fpath,
        **pr_kwargs
    )

    artists = []
    map_proj = ccrs.AlbersEqualArea(central_longitude=-90, central_latitude=38)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 4), dpi=350, subplot_kw={'projection': map_proj})
    axs = [ax1, ax2, ax3, ax4]
    for ax in axs: ax.add_feature(cfeat.STATES, linewidth=0.25)
    ax1.set_title('DL Surface P', fontsize='small')
    if cesm:
        ax2.set_title('CESM Surface P', fontsize='small')
        ax3.set_title(r'CESM $\omega$ 550hPa', fontsize='small')
        ax4.set_title(r'CESM TCWV', fontsize='small')
    else:
        ax2.set_title('MSWEPv2 Surface P', fontsize='small')
        ax3.set_title(r'MERRA2 $\omega$ 550hPa', fontsize='small')
        ax4.set_title(r'MERRA2 TCWV', fontsize='small')

    cbar_ticks = [np.arange(0, 17, 4), np.arange(0, 17, 4), np.linspace(-1, 1, 5), np.arange(0, 61, 10)]
    cbar_labels = ['[mm]', '[mm]', '[Pa/s]', r'[kg/m$^2$]']
    #area = surface_area(lsf)
    lsf['lev'] = lsf['lev'] * 100
    lsf['lev'].attrs['units'] = 'Pa'
    for i, dt in enumerate(dl_p.time):
        cq = lsf.sel(time=dt)['QV'].integrate('lev') / -9.8
        co = lsf.sel(time=dt, lev=550 * 100)['OMEGA'] 
        cp, dp = obs_p.sel(time=dt)['precip'], dl_p.sel(time=dt)['precip']
        items = utils.region_ani([dp, cp, co, cq], plot_params, ax1, ax2, ax3, ax4)
        title = plt.text(1.1, 1.3, f'TIME: {dt.values} UTC', horizontalalignment='center', verticalalignment='bottom', transform=ax2.transAxes)
        items.insert(0, title)
        artists.append(items)
        if i == 0:
            for j, (item, ax, t, l) in enumerate(zip(items[1:], axs, cbar_ticks, cbar_labels)):
                fig.colorbar(item, ax=ax, location='bottom', ticks=t, label=l)

    ani = animation.ArtistAnimation(fig, artists, interval=500)
    ani.save(filename=f'./models/{model_name}/ani.mp4', writer='ffmpeg')
    return

# ---------------------------------------------------------------------------------
def get_pdf(complete, model_name, note):
    colors = ['blue', 'black', 'red', 'green', 'gold', 'orange']
    styles = ['solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed']
    labels = ['DL', 'MSWEPv2', 'CESM2']
    labels = labels[:len(complete)]
    gp_labels = [l + '_GP' for l in labels]

    gp_hrly = [da.sel(lat=GP_SLAT, lon=GP_SLON) for da in complete]
    daily = [da.resample(time='D').sum(dim='time').dropna('time', how='all') for da in complete]
    NGP = [da.sel(lat=NGP_SLAT, lon=SLON) for da in daily]
    SGP = [da.sel(lat=SGP_SLAT, lon=SLON) for da in daily]

    ### scaled intensity plot
    thresh = np.linspace(0, 100, 100)
    P0_NGP = [_lat_wtd_mean(da.where(da > 0, 0)).mean(dim='time')['precip'].to_numpy() for da in NGP]
    P0_SGP = [_lat_wtd_mean(da.where(da > 0, 0)).mean(dim='time')['precip'].to_numpy() for da in SGP]
    print(P0_NGP, P0_SGP)
    N_ds = len(complete)
    Pr_NGP, Pr_SGP = [], []
    for t in thresh:
        out_NGP = [_lat_wtd_mean(da.where(da > t, 0)).mean(dim='time')['precip'].to_numpy() for da in NGP]
        out_SGP = [_lat_wtd_mean(da.where(da > t, 0)).mean(dim='time')['precip'].to_numpy() for da in SGP]
        Pr_NGP.append(out_NGP)
        Pr_SGP.append(out_SGP)

    Pr_NGP, Pr_SGP = np.asarray(Pr_NGP), np.asarray(Pr_SGP)
    fig, axs = plt.subplots(2, 2, layout='constrained')
    ax = axs.flatten()
    for i, l, c in zip(range(N_ds), labels, colors):
        ax[0].plot(thresh, Pr_NGP[:, i], label=l, color=c)
        ax[1].plot(thresh, Pr_NGP[:, i] / P0_NGP[i], label=l, color=c)
        ax[2].plot(thresh, Pr_SGP[:, i], label=l, color=c)
        ax[3].plot(thresh, Pr_SGP[:, i] / P0_SGP[i], label=l, color=c)
    ax[0].set(ylim=(0, 3.5), xlim=(0, 100), box_aspect=1, title='NGP')
    ax[1].set(ylim=(0, 1), xlim=(0, 100), box_aspect=1, title='NGPnorm')
    ax[2].set(ylim=(0, 3.5), xlim=(0, 100), box_aspect=1, title='SGP')
    ax[3].set(ylim=(0, 1), xlim=(0, 100), box_aspect=1, title='SGPnorm')
    plt.legend()
    plt.savefig(f'./models/{model_name}/intensity_pdf_{note}.png', dpi=300)
    plt.clf()

    ### daily pdf
    out_NGP = [da.where(da >= R_THRESH).to_dataarray().data.flatten() for da in NGP]
    out_SGP = [da.where(da >= R_THRESH).to_dataarray().data.flatten() for da in SGP]
    mask_NGP = [~np.isnan(arr) for arr in out_NGP]
    mask_SGP = [~np.isnan(arr) for arr in out_SGP]
    fin_NGP = [arr[m] for arr, m in zip(out_NGP, mask_NGP)]
    fin_SGP = [arr[m] for arr, m in zip(out_SGP, mask_SGP)]
    df_dict = {}
    for arr, lab in zip(fin_NGP + fin_SGP, ngp_labels + sgp_labels):
        print(lab, len(arr))
        df_dict[lab] = arr
    df = pd.DataFrame({k: pd.Series(v) for k, v in df_dict.items()})
    df = df.melt(var_name='models', value_name='precip')
    ax = sns.histplot(data=df, fill=False, bins=50, element='poly', x='precip', stat='probability', hue='models', log_scale=True, palette=colors, common_norm=False, multiple='dodge')
    ax.set(ylabel='Probability', xlabel='Daily Accumulated Precipitation [mm/day]')
    plt.savefig(f'./models/{model_name}/pdf_{note}.png', dpi=300)
    plt.clf()

    ### 3-hourly pdf
    out_ngp_hrly = [da.where(da >= 0.1).to_dataarray().data.flatten() for da in ngp_hrly]
    #out_SGP = [da.where(da >= -1.1).to_dataarray().data.flatten() for da in SGP]
    mask_ngp = [~np.isnan(arr) for arr in out_ngp_hrly]
    #mask_SGP = [~np.isnan(arr) for arr in out_SGP]
    fin_NGP = [arr[m] for arr, m in zip(out_ngp_hrly, mask_ngp)]
    #fin_SGP = [arr[m] for arr, m in zip(out_SGP, mask_SGP)]
    df_dict = {}
    for arr, lab in zip(fin_NGP, ngp_labels):
        print(lab, len(arr))
        df_dict[lab] = arr
    df = pd.DataFrame({k: pd.Series(v) for k, v in df_dict.items()})
    df = df.melt(var_name='models', value_name='precip')
    ax = sns.histplot(data=df, fill=False, bins=50, element='poly', x='precip', stat='probability', hue='models', log_scale=True, palette=colors, common_norm=False, multiple='dodge')
    ax.set(ylabel='Probability', xlabel='3-hourly Accumulated Precipitation [mm/3hr]')
    plt.savefig(f'./models/{model_name}/pdf_{note}_hrly.png', dpi=300)
    plt.clf()

    ## 50-99.99 percentile
    out = []
    perc = np.concat([np.arange(50, 91, 5), np.linspace(91, 99, 17), np.linspace(99.1, 99.9, 17), np.linspace(99.91, 99.99, 17)])
    for p in perc:
        out.append([np.nanpercentile(da.where(da >= R_THRESH).to_dataarray().data.flatten(), p) for da in NGP])

    out = np.asarray(out)
    n = 5
    x_int = 1 - 1 / 10 ** np.arange(n + 2)
    ticklabs = [f'{100 * v:.2f}%' for v in x_int]
    fig, ax = plt.subplots()
    for i, l, c in zip(list(range(N_ds)), ngp_labels, colors):
        ax.plot(100 - perc, out[:, i], label=l, color=c)
    ax.invert_xaxis()
    ax.set(yscale='log', xscale='log', ylim=(0.1, 100), xlabel='Percentile of distribution', ylabel='Total Rainfall (mm/day)')
    ax.xaxis.set_ticklabels(ticklabs[::-1])
    plt.legend()
    plt.savefig(f'./models/{model_name}/extremes_{note}.png', dpi=300)

# ---------------------------------------------------------------------------------
def precip(complete, plot_params, model_name, note):
    # Known Sumatra Squall events (not in test apparently)
    # selected = [(2016, 7, 12), (2014, 6, 11), (2014, 6, 12)]
    season = model_name.split('_')[1]
    match season:
        case 'MAMJJA':
            selected = ['2017-08-04', '2008-05-12', '2007-04-18', '2015-07-26']
        case 'JJA':
            #selected = ['2004-07-21', '2013-08-05', '2010-06-27', '2009-07-07']
            # animations
            #selected = ['1983-08-16', '1983-08-17', '1981-08-29', '1981-08-28', '1981-06-05', '1983-07-12']
            selected = ['2015-08-18', '2010-06-27']
        case 'MAM':
            selected = ['2010-03-17', '2009-05-10', '2016-04-10', '2013-04-25']

    complete_pred, complete_obs = complete[0], complete[-1]
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
def annual_stats(complete, plot_params, model_name, note):
    # per-year daily mean
    annual = [da.resample(time='D').sum(dim='time').dropna('time', how='all').groupby('time.year').mean(dim='time') for da in complete]

    # total annual mean daily precip
    labels = ['DL', 'CESM']
    climatology = {}
    for da, _id in zip(annual, labels):
        climatology |= {str(yr) + _id: da.sel(year=yr)['precip'].as_numpy() for yr in da.year.values}

    utils.plot_mean(
        climatology,
        plot_params,
        model_name,
        note=note + '_annual',
        bias=False
    )

    # total annual mean daily precip bias
    bias = annual[0] - annual[1]
    utils.plot_mean(
        {str(yr): bias.sel(year=yr)['precip'].as_numpy() for yr in bias.year.values},
        plot_params,
        model_name,
        note=note + '_annual',
        bias=True
    )

# ---------------------------------------------------------------------------------
def daily_stats(complete):
    # resample to regional latitude-weighted daily means
    ngp = [da.sel(lat=NGP_SLAT, lon=SLON) for da in complete]
    daily = [_lat_wtd_mean(da.resample(time='D').sum(dim='time').dropna('time', how='all'))['precip'] for da in ngp]
    #diff = (daily[0] - daily[1]).sortby(lambda x: x, ascending=False)
    min_obs = [da.sortby(lambda x: x, ascending=False) for da in daily]
    labels = ['DL', 'MSWEPv2', 'CESM2']
    for item, _id in zip(min_obs, labels):
        print(_id)
        print(item.values[:8])
        print(item.time.values[:8])

# ---------------------------------------------------------------------------------
def weekly_stats(complete):
    # resample to regional latitude-weighted daily means
    ngp = [da.sel(lat=NGP_SLAT, lon=SLON) for da in complete]
    weekly = [_lat_wtd_mean(da.resample(time='7D').sum(dim='time').dropna('time', how='all'))['precip'] for da in ngp]
    #diff = (daily[0] - daily[1]).sortby(lambda x: x, ascending=False)
    min_obs = [da.sortby(lambda x: x, ascending=False) for da in weekly]
    labels = ['DL', 'MSWEPv2', 'CESM2']
    return min_obs[0].time.values[0]

# ---------------------------------------------------------------------------------
def seasonal_stats(complete, plot_params, model_name, note):
    season = model_name.split('_')[1]
    daily = [da.resample(time='D').sum(dim='time', skipna=True).dropna('time', how='all') for da in complete]
    seasonal = [da.where(da >= R_THRESH).groupby('time.season').mean(dim='time', skipna=True) for da in daily]

    # total seasonal mean daily precip
    labels = ['DL', 'MSWEPv2', 'CESM2']
    climatology = {}
    for da, _id in zip(seasonal, labels):
        climatology |= {ssn + _id: da.sel(season=ssn)['precip'].to_numpy() for ssn in da.season.values}

    utils.plot_mean(
        climatology,
        plot_params,
        model_name,
        note=note + '_seasonal',
        bias=False
    )

    # total seasonal mean daily precip bias
    if len(seasonal) == 2:
        bias_DM = seasonal[0] - seasonal[1]
        bias = {ssn + ' DL - MSWEPv2': bias_DM.sel(season=ssn)['precip'].to_numpy() for ssn in bias_DM.season.values}
    else:
        bias_DM = seasonal[0] - seasonal[1]
        bias_CM = seasonal[2] - seasonal[1]
        bias = {ssn + ' DL - MSWEPv2': bias_DM.sel(season=ssn)['precip'].to_numpy() for ssn in bias_DM.season.values} | {ssn + ' CESM2 - MSWEPv2': bias_CM.sel(season=ssn)['precip'].to_numpy() for ssn in bias_CM.season.values}

    utils.plot_mean(
        bias,
        plot_params,
        model_name,
        note=note + '_seasonal',
        bias=True
    )

# ---------------------------------------------------------------------------------
def hourly_stats(complete, PLOT_PARAMS, model_name, note):
    season = model_name.split('_')[1]

    nsamp = [da.groupby('time.hour').count() for da in complete]
    ndays = [da.precip.values / 8 for da in nsamp]
    hourly = [da.groupby('time.hour').sum(dim='time', skipna=True) / nd for da, nd in zip(complete, ndays)]

    ### regional mean diurnal cycle line plot
    NGP = [_lat_wtd_mean(da.sel(lat=NGP_SLAT, lon=SLON).where(da >= R_THRESH))['precip'] for da in hourly]
    SGP = [_lat_wtd_mean(da.sel(lat=SGP_SLAT, lon=SLON).where(da >= R_THRESH))['precip'] for da in hourly]

    fig, [ax0, ax1] = plt.subplots(1, 2, sharey=True)
    colors = ['blue', 'black', 'red', 'green', 'gold']
    styles = ['solid', 'solid', 'solid', 'solid', 'solid']
    labels = ['DL', 'MSWEPv2', 'CESM2', 'DL4x', 'CESM4x']
    for da, _id, c in zip(NGP, labels, colors): da.plot.line(ax=ax0, color=c, label=_id)
    for da, _id, c in zip(SGP, labels, colors): da.plot.line(ax=ax1, color=c, label=_id)
    ax0.set(title='NGP', xlabel='UTC Hour', ylabel='mean regional accum. [mm/d]', xlim=(0, 21))
    ax1.set(title='SGP', xlabel='UTC Hour', ylabel='', xlim=(0, 21))
    ax0.legend(frameon=False, fancybox=False, fontsize='small')
    fig.suptitle(f'Regional Mean Diurnal Cycle for {season}')
    fig.tight_layout()
    plt.savefig(os.path.join(pth.MODEL_OUT, f'{model_name}', f'diurnal_{note}.png'), dpi=300)

    ### Hovmoller diagrams
    plt.clf()
    NGP = [_latonly_wtd_mean(da.sel(lat=NGP_SLAT, lon=SLON).where(da >= R_THRESH))['precip'] for da in hourly]
    SGP = [_latonly_wtd_mean(da.sel(lat=SGP_SLAT, lon=SLON).where(da >= R_THRESH))['precip'] for da in hourly]
    N = len(NGP)
    fig, axs = plt.subplots(2, N, sharey=True, layout='constrained')
    axs = axs.flatten()
    vmin, vmax = 0, 6
    for da, ax, _id in zip(NGP, axs[:N], labels):
        a = da.plot.contourf(ax=ax, vmin=vmin, vmax=vmax, levels=13, add_colorbar=False)
        ax.set(title=f'{_id} NGP', box_aspect=1, yticks=np.arange(0, 24, 3), xticks=np.arange(-105, -80, 5))
    for da, ax, _id in zip(SGP, axs[N:], labels):
        a = da.plot.contourf(ax=ax, vmin=vmin, vmax=vmax, levels=13, add_colorbar=False)
        ax.set(title=f'{_id} SGP', box_aspect=1, yticks=np.arange(0, 24, 3), xticks=np.arange(-105, -80, 5))
    plt.colorbar(a, ax=axs, orientation='horizontal')
    plt.savefig(os.path.join(pth.MODEL_OUT, f'{model_name}', f'hovmoller_{note}.png'), dpi=300)

# ---------------------------------------------------------------------------------
def _lat_wtd_sum(da):
    wt = np.cos(np.radians(da.lat))
    return da.weighted(wt).sum(dim=['lat', 'lon'], skipna=True)

# ---------------------------------------------------------------------------------
def _lat_wtd_mean(da):
    wt = np.cos(np.radians(da.lat))
    return da.weighted(wt).mean(dim=['lat', 'lon'], skipna=True)

# ---------------------------------------------------------------------------------
def _latonly_wtd_mean(da):
    wt = np.cos(np.radians(da.lat))
    return da.weighted(wt).mean(dim='lat', skipna=True)

# ---------------------------------------------------------------------------------
def _select_batch(ds, **kwargs):
    return ds.sel(**kwargs)

# ---------------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
