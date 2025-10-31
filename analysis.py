import xarray as xr
from dask.distributed import Client, LocalCluster
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import seaborn as sns
from functools import partial
import cftime
import pandas as pd
import geocat.comp as gc
from argparse import ArgumentParser
import os
import sys
sys.path.append('/home/eastinev/ai')
from file_utils import *
from metrics import *
from analysis_utils import *
import plot_utils as utils
import paths as pth

# Lin et al. uses 31-52N, -98--89E
GP_SLAT, GP_SLON = slice(35, 45), slice(-105, -85)
CUS_LON = slice(-110, -70)
CUS_LAT = slice(25, 51)
R_THRESH = 1  # mm/3hr

# CUS 3D Inc MERRA2 grid
GRIDSHAPE = (53, 65)  # (H, W)
EXTENT = (-110, -70, 25, 51)
LONS = np.linspace(EXTENT[0], EXTENT[1], GRIDSHAPE[1])
LATS = np.linspace(EXTENT[2], EXTENT[3], GRIDSHAPE[0])
PLOT_PARAMS = {'extent': EXTENT, 'lons': LONS, 'lats': LATS}
COLORS = ['black', 'blue', 'red']
LABELS = ['MSWEPv2', 'DL', 'CESM']

## ================================================================================
def main():
    parser = ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('-e', '--exp', type=str, default='')
    parser.add_argument('-c', '--combined', action='store_true')
    args = parser.parse_args()
    model_name = args.model_name
    exp = args.exp

    print(f'Running analysis on {model_name} w/exp {exp}')
    n_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    cluster = LocalCluster(n_workers=n_cpus)
    print(cluster)
    with Client(cluster) as client:
        print(client)
        season = model_name.split('_')[1]
        note = ''

        ## full analysis
        match exp:
            case 'CTRL' | '2K':
                dl_pall_cesm = xr.open_dataset(f'./models/{model_name}/MODEL_PRED_CESM.{exp}.nc')
                times = dl_pall_cesm.time  # already as cftime objs
                #dl_p_cesm = xr.open_dataset(f'./models/noQW_JJA_F00h/MODEL_PRED_CESM.{exp}.nc')
                cesm_p = xr.open_dataset(f'./CESM.{exp}_1979-1983.nc').sel(time=times).rename({'PRECT': 'precip'})
                mswep_p_cesm = xr.open_dataset(f'./MSWEP_1979-1983.nc').convert_calendar('noleap', use_cftime=True).sel(time=times)
                #dl_pall_cesm2 = xr.open_dataset(f'./models/{model_name}/MODEL_PRED_CESM.2K.nc')
                #times = dl_pall_cesm.time  # already as cftime objs
                #dl_p_cesm2 = xr.open_dataset(f'./models/noQW_JJA_F00h/MODEL_PRED_CESM.2K.nc')
                #cesm_p2 = xr.open_dataset(f'./CESM.2K_1979-1983.nc').sel(time=times).rename({'PRECT': 'precip'})
                #complete = [dl_pall_cesm2, dl_pall_cesm, dl_p_cesm2, dl_p_cesm, cesm_p2, cesm_p]
                complete = [mswep_p_cesm, dl_pall_cesm, cesm_p]
                note = f'CESM.{exp}'
                cesm = True
            case 'all':
                dl_pall_test = xr.open_dataset(f'./models/{model_name}/MODEL_PRED_all.nc')
                #dl_p_test = xr.open_dataset(f'./models/noQW_JJA_F00h/MODEL_PRED_all.nc')
                mswep_p_test = xr.open_dataset(f'./models/{model_name}/MSWEP_all.nc')
                complete = [mswep_p_test, dl_pall_test]
                #complete = [dl_pall_test, dl_p_test, mswep_p_test]
                note = 'all_ctrl'
                cesm = False
            case '':
                dl_pall_test = xr.open_dataset(f'./models/{model_name}/MODEL_PRED.nc')
                #dl_p_test = xr.open_dataset(f'./models/noQW_JJA_F00h/MODEL_PRED.nc')
                mswep_p_test = xr.open_dataset(f'./models/{model_name}/MSWEP.nc')
                complete = [mswep_p_test, dl_pall_test]
                #complete = [dl_pall_test, dl_p_test, mswep_p_test]
                note = 'ctrl'
                cesm = False

        ## filter for R threshold before doing processing
        complete = [da.where(da >= R_THRESH) for da in complete]
        get_pdf(complete, model_name, note)
        return
        time_slice = np.datetime64(daily_stats(complete))
        precip(complete, time_slice.astype(str), PLOT_PARAMS, model_name, note, cesm=cesm)
        seasonal_stats(complete, PLOT_PARAMS, model_name, note)
        #annual_stats(complete, PLOT_PARAMS, model_name, note)
        #daily_stats(complete)
        diurnal(complete, PLOT_PARAMS, model_name, note)
        hourly_stats(complete, PLOT_PARAMS, model_name, note)

        ## animations
        #t = np.datetime64(daily_stats(complete)) - np.timedelta64(4, 'D')
        #tend = t + np.timedelta64(3, 'D')
        #time = slice(t.astype(str).split('T')[0], tend.astype(str).split('T')[0])
        #animate(model_name, PLOT_PARAMS, time, cesm=cesm)
        #animate_precip(model_name, PLOT_PARAMS, time)

# ---------------------------------------------------------------------------------
def animate(model_name, plot_params, time, cesm=True, exp=''):
    tsel = time.start[:7].replace('-', '_')
    kwargs = {'lat': CUS_LAT, 'lon': CUS_LON, 'time': time}
    pr_kwargs = {'drop_vars': [], 'time': time}
    if cesm:
        lsf_fpath = os.path.join(pth.SCRATCH, 'cus_cesm', f'LSF_{tsel}*.{exp}.nc')
        obs_p_fpath = os.path.join('CESM.{exp}_1979-1983.nc'),
        dl_p_fpath = os.path.join('models', model_name, 'MODEL_PRED_CESM.{exp}.nc')
        kwargs |= {'drop_vars': ['T', 'U', 'V', 'Z3']}
    else:
        lsf_fpath = os.path.join(pth.SCRATCH, 'cus', f'LSF_{tsel}*.nc')
        kwargs |= {'drop_vars': ['T', 'U', 'V', 'H']}
        obs_p_fpath = os.path.join('models', model_name, f'MSWEP_all.nc')
        dl_p_fpath = os.path.join('models', model_name, 'MODEL_PRED_all.nc')

    lsf = open_mf(
        lsf_fpath,
        **kwargs
    )
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
        # TODO: this correct?
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
def animate_precip(model_name, plot_params, time):
    tsel = time.start[:7].replace('-', '_')
    pr_kwargs = {'drop_vars': [], 'time': time}
    obs_p_fpath = os.path.join('models', model_name, f'MSWEP_all.nc')
    dl_pall_fpath = os.path.join('models', model_name, 'MODEL_PRED_all.nc')
    dl_p_fpath = os.path.join('models', 'noQW_JJA_F00h', 'MODEL_PRED_all.nc')

    obs_p = open_mf(
        obs_p_fpath,
        **pr_kwargs
    )
    dl_pall = open_mf(
        dl_pall_fpath,
        **pr_kwargs
    )
    dl_p = open_mf(
        dl_p_fpath,
        **pr_kwargs
    )
    artists = []
    map_proj = ccrs.AlbersEqualArea(central_longitude=-90, central_latitude=38)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4), dpi=350, subplot_kw={'projection': map_proj})
    axs = [ax1, ax2, ax3]
    for ax in axs: ax.add_feature(cfeat.STATES, linewidth=0.25)
    ax1.set_title('DL All Surface P', fontsize='small')
    ax2.set_title('DL noQW Surface P', fontsize='small')
    ax3.set_title('MSWEPv2 Surface P', fontsize='small')
    cbar_ticks = np.arange(0, 17, 4)
    cbar_label = '[mm]'
    for i, dt in enumerate(dl_p.time):
        cp, dp, dpall = obs_p.sel(time=dt)['precip'], dl_p.sel(time=dt)['precip'], dl_pall.sel(time=dt)['precip']
        items = utils.precip_ani([dpall, dp, cp], plot_params, ax1, ax2, ax3)
        title = plt.text(0.5, 1.3, f'TIME: {str(dt.values)[:-10]} UTC', horizontalalignment='center', verticalalignment='bottom', transform=ax2.transAxes)
        items.insert(0, title)
        artists.append(items)
        if i == 0:
            fig.colorbar(items[-1], ax=axs, location='bottom', ticks=cbar_ticks, label=cbar_label, shrink=0.35)

    ani = animation.ArtistAnimation(fig, artists, interval=500)
    ani.save(filename=f'./models/precip_ani.mp4', writer='ffmpeg')

# ---------------------------------------------------------------------------------
def get_pdf(complete, model_name, note):
    daily = [da.resample(time='D').sum(dim='time', skipna=True).dropna('time', how='all') for da in complete]
    GP = [da.sel(lat=GP_SLAT, lon=GP_SLON) for da in daily]
    GP_hrly = [da.sel(lat=GP_SLAT, lon=GP_SLON) for da in complete]

    ### scaled intensity plot
    thresh = np.linspace(R_THRESH, 100, 100)
    P0_GP = [lat_wtd_mean(da.where(da >= R_THRESH, 0)).mean(dim='time')['precip'].to_numpy() for da in GP]
    print(P0_GP)
    N_ds = len(complete)
    Pr_GP = []
    for t in thresh:
        out_GP = [lat_wtd_mean(da.where(da >= t, 0)).mean(dim='time')['precip'].to_numpy() for da in GP]
        Pr_GP.append(out_GP)

    Pr_GP = np.asarray(Pr_GP)
    fig, axs = plt.subplots(1, 2, layout='constrained')
    ax = axs.flatten()
    for i, l, c in zip(range(N_ds), LABELS, COLORS):
        ax[0].plot(thresh, Pr_GP[:, i], label=l, color=c)
        ax[1].plot(thresh, Pr_GP[:, i] / P0_GP[i], label=l, color=c)
    ax[0].set(ylim=(0, 3.5), xlim=(0, 100), box_aspect=1, title='GP')
    ax[1].set(ylim=(0, 1), xlim=(0, 100), box_aspect=1, title='GPnorm')
    plt.legend()
    plt.savefig(f'./models/{model_name}/intensity_pdf_{note}.png', dpi=300)
    plt.clf()

    ### 3-hourly pdf
    out_GP = [da.to_dataarray().data.flatten() for da in GP_hrly]
    mask_GP = [~np.isnan(arr) for arr in out_GP]
    fin_GP = [arr[m] for arr, m in zip(out_GP, mask_GP)]
    df_dict = {}
    for arr, lab in zip(fin_GP, LABELS):
        print(lab, len(arr))
        df_dict[lab] = arr
    df = pd.DataFrame({k: pd.Series(v) for k, v in df_dict.items()})
    df = df.melt(var_name='models', value_name='precip')
    ax = sns.histplot(data=df, fill=False, bins=50, element='poly', x='precip', stat='probability', hue='models', log_scale=True, palette=COLORS, common_norm=False, multiple='dodge')
    ax.set(ylabel='Probability', xlabel='Daily Accumulated Precipitation [mm/day]')
    plt.savefig(f'./models/{model_name}/h3_pdf_{note}.png', dpi=300)
    plt.clf()

    ### daily pdf
    out_GP = [da.to_dataarray().data.flatten() for da in GP]
    mask_GP = [~np.isnan(arr) for arr in out_GP]
    fin_GP = [arr[m] for arr, m in zip(out_GP, mask_GP)]
    df_dict = {}
    for arr, lab in zip(fin_GP, LABELS):
        print(lab, len(arr))
        df_dict[lab] = arr
    df = pd.DataFrame({k: pd.Series(v) for k, v in df_dict.items()})
    df = df.melt(var_name='models', value_name='precip')
    ax = sns.histplot(data=df, fill=False, bins=50, element='poly', x='precip', stat='probability', hue='models', log_scale=True, palette=COLORS, common_norm=False, multiple='dodge')
    ax.set(ylabel='Probability', xlabel='Daily Accumulated Precipitation [mm/day]')
    plt.savefig(f'./models/{model_name}/d1_pdf_{note}.png', dpi=300)
    plt.clf()

    ## 50-99.99 percentile
    out = []
    perc = np.concat([np.arange(50, 91, 5), np.linspace(91, 99, 17), np.linspace(99.1, 99.9, 17), np.linspace(99.91, 99.99, 17)])
    for p in perc:
        out.append([np.nanpercentile(da.to_dataarray().data.flatten(), p) for da in GP])

    out = np.asarray(out)
    n = 5
    x_int = 1 - 1 / 10 ** np.arange(n + 2)
    ticklabs = [f'{100 * v:.2f}%' for v in x_int]
    fig, ax = plt.subplots()
    for i, l, c in zip(list(range(N_ds)), LABELS, COLORS):
        ax.plot(100 - perc, out[:, i], label=l, color=c)
    ax.invert_xaxis()
    ax.set(yscale='log', xscale='log', ylim=(1, 200), xlabel='Percentile of distribution', ylabel='Total Rainfall (mm/day)')
    ax.xaxis.set_ticklabels(ticklabs[::-1])
    plt.legend()
    plt.savefig(f'./models/{model_name}/extremes_{note}.png', dpi=300)

# ---------------------------------------------------------------------------------
def precip(complete, time_slice, plot_params, model_name, note, cesm=False):
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

    if cesm:
        complete_obs, complete_pred = complete[2], complete[1]
    else:
        complete_obs, complete_pred = complete[0], complete[1]
    selected = [time_slice.split('T')[0]]
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
    climatology = {}
    for da, _id in zip(annual, LABELS):
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
    gp = [da.sel(lat=GP_SLAT, lon=GP_SLON) for da in complete]
    daily = [lat_wtd_mean(da.resample(time='D').sum(dim='time').dropna('time', how='all'))['precip'] for da in gp]
    #diff = (daily[0] - daily[1]).sortby(lambda x: x, ascending=False)
    min_obs = [da.sortby(lambda x: x, ascending=False) for da in daily]
    return min_obs[1].time.values[0]

# ---------------------------------------------------------------------------------
def weekly_stats(complete):
    # resample to regional latitude-weighted daily means
    ngp = [da.sel(lat=NGP_SLAT, lon=SLON) for da in complete]
    weekly = [lat_wtd_mean(da.resample(time='7D').sum(dim='time').dropna('time', how='all'))['precip'] for da in ngp]
    #diff = (daily[0] - daily[1]).sortby(lambda x: x, ascending=False)
    min_obs = [da.sortby(lambda x: x, ascending=False) for da in weekly]
    return min_obs[1].time.values[0]

# ---------------------------------------------------------------------------------
def seasonal_stats(complete, plot_params, model_name, note):
    season = model_name.split('_')[1]
    daily = [da.resample(time='D').sum(dim='time', skipna=True).dropna('time', how='all') for da in complete]
    seasonal = [da.groupby('time.season').mean(dim='time', skipna=True) for da in daily]

    # total seasonal mean daily precip
    climatology = {}
    for da, _id in zip(seasonal, LABELS):
        climatology |= {_id: da.sel(season=ssn)['precip'].to_numpy() for ssn in da.season.values}

    utils.plot_mean(
        climatology,
        plot_params,
        model_name,
        note=note + '_seasonal',
        bias=False
    )

    # total seasonal mean daily precip bias
    bias = {}
    for da, _id in zip(seasonal[1:], LABELS[1:]):
        bias |= {_id + ' - MSWEPv2': da.sel(season=ssn)['precip'].to_numpy() - seasonal[0].sel(season=ssn)['precip'].to_numpy() for ssn in da.season.values}
    utils.plot_mean(
        bias,
        plot_params,
        model_name,
        note=note + '_seasonal',
        bias=True
    )

# ---------------------------------------------------------------------------------
def diurnal(complete, plot_params, model_name, note):
    # mm/month... but want in mm/day.? assume 30 day months?
    # resample to monthly
    mmean = [da.resample(time='ME').sum(dim='time', skipna=True).dropna('time', how='all').mean(dim='time') / 30 for da in complete]
    #GP = [lat_wtd_mean(da.sel(lat=GP_SLAT, lon=GP_SLON))['precip'] for da in complete]
    #reg_mean = [da.groupby('time.hour').mean(dim='time', skipna=True).rename({'hour': 'time'}) * 8 for da in GP]
    # NOTE: apparently these operations commute
    hourly = [da.groupby('time.hour').mean(dim='time', skipna=True).rename({'hour': 'time'}) * 8 for da in complete]
    reg_mean = [lat_wtd_mean(da.sel(lat=GP_SLAT, lon=GP_SLON))['precip'] for da in hourly]
    #hourly = complete
    # lat, lon, hour --> fft over hour (subtract mean?)
    def fft_fn(x):
        return np.fft.fft(x)

    phase_out = []
    dial_out = []
    for da_h, da_m in zip(hourly, mmean):
        fft_out = xr.apply_ufunc(
            fft_fn,
            da_h - da_h.mean(dim='time'),
            input_core_dims=[['time']],
            output_core_dims=[['frequency']],
            exclude_dims=set(('time',)),
            dask='parallelized',
            vectorize=True
        )
        n_time = da_h.sizes['time']
        freqs = np.fft.fftfreq(n_time, d=1/8)  # sample rate is 1 samp / 3 hours --> 1 samp / (1/8) day
        fft_out = fft_out.assign_coords(frequency=freqs)
        diu = fft_out.isel(frequency=1)
        ## NOTE: phase of 0 ~ UTC00/24, phase of -pi ~UTC12, phase of -pi/2 ~UTC06
        amp, phs = np.abs(diu), xr.ufuncs.angle(diu)
        out = xr.where((amp / da_m < 0.25) | (da_m < 0.75), np.nan, phs)
        ## shift phase and map to 24hr clock?
        phase_out.append(out['precip'].to_numpy())

    titles = LABELS
    utils.region_plot(
        phase_out,
        PLOT_PARAMS | {'cmap': 'phase'},
        titles,
        '',
        f'./models/{model_name}/phase_{note}.png',
        figsize=(12, 4)
    )

    ## harmonic dial plot; take this spatially-explicit mean diurnal cycle and average across gridpoints
    fig, ax = plt.subplots(layout='constrained', subplot_kw={'projection': 'polar'})
    ax.set_rticks([1, 2, 3])
    for da, l, c in zip(reg_mean, LABELS, COLORS):
        fft_out = xr.apply_ufunc(
            fft_fn,
            da - da.mean(dim='time'),
            input_core_dims=[['time']],
            output_core_dims=[['frequency']],
            exclude_dims=set(('time',)),
            dask='parallelized',
            vectorize=True
        )
        n_time = da.sizes['time']
        # sample rate is 1 samp / 3 hours --> 1 samp / (1/8) day
        freqs = np.fft.fftfreq(n_time, d=1/8)  # in cycles/day
        fft_out = fft_out.assign_coords(frequency=freqs)
        diu = fft_out.isel(frequency=1)
        amp, phs = np.abs(diu), xr.ufuncs.angle(diu)
        ax.plot(phs, amp, 'o', color=c, label=l)
        ax.plot([0, phs], [0, amp], '-', color=c)

    ax.grid(True)
    ax.legend()
    plt.savefig(f'./models/{model_name}/dial_{note}.png', dpi=300)

    return

# ---------------------------------------------------------------------------------
def hourly_stats(complete, plot_params, model_name, note):
    season = model_name.split('_')[1]

    #nsamp = [da.groupby('time.hour').count() for da in complete]
    #ndays = [da.precip.values / 8 for da in nsamp]
    #hourly = [da.groupby('time.hour').sum(dim='time', skipna=True) / nd for da, nd in zip(complete, ndays)]
    ## is this actually the more appropriate way.. just multiply by 8.?
    #hourly = [da.groupby('time.hour').mean(dim='time', skipna=True) * 8 for da in complete]
    hourly = [da.where(da >= R_THRESH) for da in complete]

    ### regional mean diurnal cycle line plot
    GP = [lat_wtd_mean(da.sel(lat=GP_SLAT, lon=GP_SLON))['precip'] for da in hourly]
    GP = [da.groupby('time.hour').mean(dim='time', skipna=True) * 8 for da in GP]

    fig, ax = plt.subplots(1, 1, layout='constrained')
    for da, _id, c in zip(GP, LABELS, COLORS): da.plot.line(ax=ax, color=c, label=_id)
    ax.set(xlabel='UTC Hour', ylabel='Precipitation [mm/d]', xlim=(0, 21), xticks=np.arange(0, 22, 3))
    ax.legend(frameon=False, fancybox=False, fontsize='small')
    plt.savefig(os.path.join(pth.MODEL_OUT, f'{model_name}', f'diurnal_{note}.png'), dpi=300)

    ### Hovmoller diagrams
    plt.clf()
    GP = [latonly_wtd_mean(da.sel(lat=GP_SLAT, lon=GP_SLON))['precip'] for da in hourly]
    GP = [da.groupby('time.hour').mean(dim='time', skipna=True) * 8 for da in GP]
    N = len(GP)
    fig, axs = plt.subplots(1, N, figsize=(12, 6), sharey=True, layout='constrained')
    axs = axs.flatten()
    cmap = cm.YlGnBu
    vmin, vmax = 0, max([da.max().values for da in GP])
    print(vmax)
    #cmap = cm.BrBG
    #vmin, vmax = -3, 3
    for da, ax, _id in zip(GP, axs[:N], LABELS):
        a = da.plot.contourf(ax=ax, vmin=vmin, vmax=vmax, levels=13, extend='max', cmap=cmap, add_colorbar=False)
        #a = da.plot.contourf(ax=ax, vmin=vmin, vmax=vmax, levels=13, extend='both', cmap='BrBG', add_colorbar=False)
        ax.set(title=f'{_id}', box_aspect=1, yticks=np.arange(0, 24, 3), xticks=np.arange(-105, -80, 5), ylabel='', xlabel='Longitude')
    axs[0].set(ylabel='UTC Hour')
    fig.colorbar(a, ax=axs, location='bottom', label='[mm/d]', shrink=0.4)
    plt.savefig(os.path.join(pth.MODEL_OUT, f'{model_name}', f'hovmoller_{note}.png'), dpi=300)

# ---------------------------------------------------------------------------------
## O'Gorman scaling stuff
## open and process LSF
def process_scaling(model_name, exp):
    lsf_path = os.path.join(pth.SCRATCH, 'cus_cesm', f'LSF*.{exp}.nc')
    lsf = open_mf(lsf_path, drop_vars=['U', 'V', 'Z3'], lat=CUS_LAT, lon=CUS_LON)
    lsf_daily = lsf.resample(time='D').mean(dim='time').dropna('time', how='all')
    lsf_daily = lsf_daily.sel(time=lsf_daily.time.dt.month.isin([6, 7, 8]))

    ## open and process CESM Pr --> do same for DL
    #pr = xr.open_dataset(f'./CESM.{exp}_1979-1983.nc')
    pr = xr.open_dataset(f'./models/{model_name}/MODEL_PRED_CESM.{exp}.nc')
    pr = pr.rename({'precip': 'PRECT'})
    Rx1day = pr.resample(time='D').sum(dim='time').dropna('time', how='all')
    Rx1day = Rx1day.sel(time=Rx1day.time.dt.month.isin([6, 7, 8]))
    tall = Rx1day.expand_dims(dim={'plev': lsf_daily.plev}).transpose('time', 'plev', ...)

    ps_path = os.path.join(pth.SCRATCH, 'cus_cesm', f'PS*.{exp}.nc')
    ps = open_mf(ps_path, lat=CUS_LAT, lon=CUS_LON)
    ps = ps.resample(time='D').mean(dim='time').dropna('time', how='all')
    ps = ps.sel(time=ps.time.dt.month.isin([6, 7, 8]))
    ps_tall = ps.expand_dims(dim={'plev': lsf_daily.plev}).transpose('time', 'plev', ...)

    comb = xr.merge([lsf_daily, tall, ps_tall])

    yr_out = []
    for yr in range(1979, 1984):
        a = comb.sel(time=str(yr))['PRECT'].idxmax(dim='time')
        yr_out.append(comb.sel(time=a))
    yr_dim = xr.DataArray(list(range(1979, 1984)), dims='year', name='year')
    max_all = xr.concat(yr_out, dim=yr_dim)
    max_all.to_netcdf(f'./models/{model_name}/CESM.{exp}_DL_scaling.nc', engine='netcdf4')

    scaling = xr.open_dataset(f'./models/{model_name}/CESM.{exp}_DL_scaling.nc')
    scaling['plev'] = scaling['plev'] * 100
    scaling['plev'].attrs['units'] = 'Pa'
    scaling['QS'] = utils.calc_qsat(scaling['T'], scaling['plev'])
    scaling['diffQS'] = scaling['QS'].differentiate('plev')  # kg_v/(kg_a*Pa)
    scaling = scaling.drop_vars('time')
    scaling = scaling.rename({'plev': 'lev'})
    g = -9.80665
    # NOTE: this is crazy slow.. can just do by hand.?
    weight = gc.meteorology.delta_pressure(pressure_lev=scaling.lev, surface_pressure=scaling.PS.isel(lev=0)) / g
    # filter for regions of ascent and below 50 hPa w/lapse rate of 2 K/km, just using 50 hPa thresh rn
    filtered = scaling.sel(lev=slice(100000, 5000))[['OMEGA', 'diffQS']].where(scaling.OMEGA < 0)
    scaling_P = filtered['OMEGA'] * filtered['diffQS'] * weight  # kg_v/(m2 * s)
    sec_per_day = 3600 * 24
    out = scaling_P.sum(dim='lev', skipna=True) * sec_per_day
    out.to_netcdf(f'./models/{model_name}/CESM.{exp}_DL_scalingP.nc', engine='netcdf4')
    return

# ---------------------------------------------------------------------------------
def plot_scaling(model_name, exp):
    ## MSWEPv2 Rx1day
    mswep_Rx1day_pr = xr.open_dataset(f'./MSWEP_1979-1983.nc')
    mswep_Rx1day_pr = mswep_Rx1day_pr.resample(time='D').sum(dim='time').dropna('time', how='all')
    mswep_Rx1day_pr = mswep_Rx1day_pr.sel(time=mswep_Rx1day_pr.time.dt.month.isin([6, 7, 8]))
    mswep_Rx1day_pr = mswep_Rx1day_pr.groupby('time.year').max(dim='time')
    mswep_Rx1day_pr = mswep_Rx1day_pr.mean(dim='year')
    mswepr_data = mswep_Rx1day_pr['precip']
    mswep_Rx1day_pr_data = {r'MSWEPv2 Rx1day': mswepr_data.to_numpy()}

    ## CESM Rx1day
    cesm_Rx1day_pr = xr.open_dataset(f'./CESM.{exp}_1979-1983.nc')
    cesm_Rx1day_pr = cesm_Rx1day_pr.resample(time='D').sum(dim='time').dropna('time', how='all')
    cesm_Rx1day_pr = cesm_Rx1day_pr.sel(time=cesm_Rx1day_pr.time.dt.month.isin([6, 7, 8]))
    cesm_Rx1day_pr = cesm_Rx1day_pr.groupby('time.year').max(dim='time')
    cesm_Rx1day_pr = cesm_Rx1day_pr.mean(dim='year')
    cesmr_data = cesm_Rx1day_pr['PRECT']
    cesm_Rx1day_pr_data = {r'CESM Rx1day': cesmr_data.to_numpy()}
    
    ## CESM Rx1day from scaling argument
    cesm_scaling_pr = xr.open_dataset(f'./CESM.{exp}_scalingP.nc')
    cesm_scaling_pr = cesm_scaling_pr.mean(dim='year')
    cesm_scaling_pr = cesm_scaling_pr.rename({'__xarray_dataarray_variable__': 'precip'})
    cesms_data = cesm_scaling_pr['precip']
    cesm_scaling_pr_data = {r'CESM Scaling $P_e$': cesms_data.to_numpy()}

    ## DL model Rx1day
    dl_Rx1day_pr = xr.open_dataset(f'./models/{model_name}/MODEL_PRED_CESM.{exp}.nc')
    dl_Rx1day_pr = dl_Rx1day_pr.resample(time='D').sum(dim='time').dropna('time', how='all')
    dl_Rx1day_pr = dl_Rx1day_pr.sel(time=dl_Rx1day_pr.time.dt.month.isin([6, 7, 8]))
    dl_Rx1day_pr = dl_Rx1day_pr.groupby('time.year').max(dim='time')
    dl_Rx1day_pr = dl_Rx1day_pr.mean(dim='year')
    dlr_data = dl_Rx1day_pr['precip']
    dl_Rx1day_pr_data = {r'DL Rx1day': dlr_data.to_numpy()}

    ## DL model Rx1day from scaling argument
    dl_scaling_pr = xr.open_dataset(f'./models/{model_name}/CESM.{exp}_DL_scalingP.nc')
    dl_scaling_pr = dl_scaling_pr.mean(dim='year')
    dl_scaling_pr = dl_scaling_pr.rename({'__xarray_dataarray_variable__': 'precip'})
    dls_data = dl_scaling_pr['precip']
    dl_scaling_pr_data = {r'DL Scaling $P_e$': dls_data.to_numpy()}

    #data = cesm_Rx1day_pr_data | cesm_scaling_pr_data | dl_Rx1day_pr_data | dl_scaling_pr_data
    #datas = data.values()
    #titles = data.keys()
    datas = [dls_data.to_numpy() / dlr_data.to_numpy()]
    titles = ['DL Scaling : DL Rx1day']
    utils.region_plot(datas, PLOT_PARAMS | {'cmap': 'ratio'}, titles, 'a', './dl_noqw_ratio.png', figsize=(12, 4))
    return
    utils.plot_mean(
        data,
        PLOT_PARAMS,
        model_name,
        note=note + f'_scaling_{exp}',
        bias=False
    )
    return

# ---------------------------------------------------------------------------------
def get_perK_scaling(): 
    ### annual regional-mean near-surface temperature [[299.46143894 298.58943646 300.56765913 299.34532356 298.54942878]]
    ### historically-forced regional-mean near-surface temperature 295.88 K
    ### this for per-gridcell linear regression of fractional change in Rx1day against mean surface temp
    cesm_ctrl = xr.open_dataset(f'./CESM.CTRL_scalingP.nc')
    cesm_ctrl = cesm_ctrl.rename({'__xarray_dataarray_variable__': 'precip'})
    cesm_ctrl = cesm_ctrl.mean('year')
    cesm_frac = (cesm_scaling_pr - cesm_ctrl) / cesm_ctrl * 100  # fractional change in %
    adder = xr.full_like(cesm_frac.sel(year=[1979]), 0)
    adder['year'] = ('year', [1978])
    cesm_frac = xr.merge([cesm_frac, adder])
    cesm_frac['year'] = ('year', np.array([295.88, 299.46143894, 298.58943646, 300.56765913, 299.34532356, 298.54942878]) - 295.88)
    out = cesm_frac.polyfit(dim='year', deg=1)
    fig, ax = plt.subplots()
    out.sel(degree=1, drop=True)['precip_polyfit_coefficients'].plot.contourf(ax=ax, cmap='Spectral', vmin=-18, vmax=18, levels=13, extend='both')
    plt.savefig('./test.png', dpi=300)
    return

# ---------------------------------------------------------------------------------
def get_Tsurf(exp):
    ## get surface temp for comparing with change
    T = open_mf(
        os.path.join(pth.SCRATCH, 'cus_cesm', f'LSF_*.{exp}.nc'),
        drop_vars=['U', 'V', 'OMEGA', 'Z3', 'Q'],
        lat=CUS_LAT,
        lon=CUS_LON
    )
    T = T.resample(time='D').mean(dim='time').dropna('time', how='all')
    T = T.sel(time=T.time.dt.month.isin([6, 7, 8]))
    PS = open_mf(
        os.path.join(pth.SCRATCH, 'cus_cesm', f'PS_*.{exp}.nc'),
        drop_vars=[],
        lat=CUS_LAT,
        lon=CUS_LON
    )
    PS = PS.resample(time='D').mean(dim='time').dropna('time', how='all')
    PS = PS.sel(time=PS.time.dt.month.isin([6, 7, 8]))
    T['plev'] = T['plev'] * 100
    TS = T.sel(plev=PS['PS'], method='nearest')

    ## get average near-surface temp.?
    a = TS.groupby('time.year').mean(dim='time')
    b = a.weighted(surface_area(a)).mean(dim=['lat', 'lon'])
    print(b.to_dataarray().values)
    return

# ---------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
