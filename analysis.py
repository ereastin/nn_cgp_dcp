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
THRESH = 0.

# CUS 3D Inc MERRA2 grid
GRIDSHAPE = (53, 65)  # (H, W)
EXTENT = (-110, -70, 25, 51)
LONS = np.linspace(EXTENT[0], EXTENT[1], GRIDSHAPE[1])
LATS = np.linspace(EXTENT[2], EXTENT[3], GRIDSHAPE[0])
PLOT_PARAMS = {'extent': EXTENT, 'lons': LONS, 'lats': LATS}
COLORS = ['black', 'blue', 'red']
LABELS = ['MSWEPv2', 'DL', 'CESM']
PDF_KWARGS = {
    'binwidth': 0.1,  # this bin width (in mm/d) used in Ma et al 2022; they also use precip range from 10^-2 to 10^3
    'fill': False,
    'element': 'poly',
    'x': 'precip',
    'stat': 'percent',  # instead of probability?
    'hue': 'models',
    'log_scale': (True, True),
    'palette': COLORS,
    'common_norm': False,
    'multiple': 'dodge'
}
# TODO: for plotting can probably create a PlotContext class or similar that holds everything.?

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
                three_hourly = [mswep_p_cesm, dl_pall_cesm, cesm_p]
                note = f'CESM.{exp}'
                cesm = True
            case 'all':
                dl_pall_test = xr.open_dataset(f'./models/{model_name}/MODEL_PRED_all.nc')
                #dl_p_test = xr.open_dataset(f'./models/noQW_JJA_F00h/MODEL_PRED_all.nc')
                mswep_p_test = xr.open_dataset(f'./models/{model_name}/MSWEP_all.nc')
                three_hourly = [mswep_p_test, dl_pall_test]
                #complete = [dl_pall_test, dl_p_test, mswep_p_test]
                note = 'all_ctrl'
                cesm = False
            case '':
                dl_pall_test = xr.open_dataset(f'./models/{model_name}/MODEL_PRED.nc')
                #dl_p_test = xr.open_dataset(f'./models/noQW_JJA_F00h/MODEL_PRED.nc')
                mswep_p_test = xr.open_dataset(f'./models/{model_name}/MSWEP.nc')
                three_hourly = [mswep_p_test, dl_pall_test]
                #complete = [dl_pall_test, dl_p_test, mswep_p_test]
                note = 'ctrl'
                cesm = False

        note += f'_T{THRESH}'

        #process_scaling(model_name, exp, daily=False, cesm=False)
        #plot_scaling(model_name, exp, daily=False)
        #return

        # another NOTE: daily statistics dont make a ton of sense for the MCS-filtered test data
        def select_great_plains(data: list[xr.DataArray]) -> list[xr.DataArray]:
            return [da.sel(lat=GP_SLAT, lon=GP_SLON) for da in data]

        def threshold(data: list[xr.DataArray], threshold_value: float, other: float = np.nan) -> list[xr.DataArray]:
            return [da.where(da >= threshold_value, other) for da in data]

        three_hourly = threshold(three_hourly, 0.0, 0.0)  # this because DL models will predict negative values
        daily = [da.resample(time='D').sum(dim='time').dropna('time', how='all') for da in three_hourly]

        print(three_hourly.groupby('time.hour').count())
        return

        intermittency(three_hourly, model_name, note)

        ## select just precipitating days/3hours
        daily_wet = threshold(daily, 1.0)  # want >= 1 mm/d to count as wet day
        three_hourly_wet = threshold(three_hourly, 0.3)   # common to use 0.1 mm/hr so 0.3 mm/3hr?

        ## seasonal daily mean
        seasonal_daily_mean = [da.groupby('time.season').mean(dim='time', skipna=True) for da in daily]

        # NOTE: which of any of these even makes sense to threshold data beforehand.?
        plot_extremes(select_great_plains(three_hourly), model_name, note)
        plot_extremes(select_great_plains(daily), model_name, note)
        plot_rx(three_hourly, model_name, note)
        plot_rx(daily, model_name, note)
        plot_pdf(select_great_plains(three_hourly), model_name, note)
        plot_pdf(select_great_plains(daily), model_name, note)
        ##
        time_slice = np.datetime64(daily_stats(three_hourly)).astype(str)
        plot_precip(three_hourly, time_slice, model_name, note, cesm=cesm)
        plot_seasonal_stats(seasonal_daily_mean, model_name, note)
        diurnal_line_plot(select_great_plains(three_hourly), model_name, note)
        hovmoller_plot(select_great_plains(three_hourly), model_name, note)
        spatial_phase_plot(three_hourly, model_name, note)
        dial_plot(select_great_plains(three_hourly), model_name, note)

        ## animations
        #t = np.datetime64(daily_stats(complete)) - np.timedelta64(4, 'D')
        #tend = t + np.timedelta64(3, 'D')
        #time = slice(t.astype(str).split('T')[0], tend.astype(str).split('T')[0])
        #animate(model_name, PLOT_PARAMS, time, cesm=cesm)
        #animate_precip(model_name, PLOT_PARAMS, time)

# ---------------------------------------------------------------------------------
def daily_metrics(daily_data: list[xr.DataArray]) -> None:
    ## NOTE(ereastin): ig why even use these? if the point is DCP and timing of intensity then who gives a shit
    """
    Common precip extreme indices:
        Rx1day/Rx5day: Annual (or seasonal) maximum single- or 5-day precipitation total. Common to take mean across years
        R95p/R99p: Very/extremely wet days, total precip for days above 95th or 99th percentile (distribution definition dep.)
        RNNmm: number of days with >= NN mm/d
        PRCPTOT: total (annual/seasonal/other) rainfall on wet days
        SDII: mean PRCPTOT
    """
    sdii = [da.where(da >= 1.).mean(dim='time', skipna=True) for da in daily]  # mean precip on wet days
    prcptot = [da.where(da >= 1.).sum(dim='time', skipna=True) for da in daily]  # total precipitation on wet days
    r1mm = [da.where(da >= 1.).count(dim='time', skipna=True) for da in daily]  # total number of wet days
    r10mm = [da.where(da >= 10.).count(dim='time', skipna=True) for da in daily]  # total number of heavy precip days
    r20mm = [da.where(da >= 20.).count(dim='time', skipna=True) for da in daily]  # total number of very heavy precip days

# ---------------------------------------------------------------------------------
def intermittency(three_hourly_data: list[xr.DataArray], model_name: str, note: str) -> None:
    ## See Covey et al. (2018); just look at seasonal for now.?
    #june = 
    #july = 
    #august = 
    mean_dcp = [da.groupby('time.hour').mean('time') for da in three_hourly_data]  # dims: lat, lon, hour
    daily_mean = [da.resample(time='D').mean('time') for da in three_hourly_data]  # dims: lat, lon, day
    seasonal_mean = [da.mean('time') for da in daily_mean]   # this is composite seasonal mean
    ## just making sure these are in fact the same
    #seasonal_mean0 = [da.mean('hour') for da in mean_diurnal_cycle]
    #np.testing.assert_allclose(seasonal_mean[0]['precip'].to_numpy(), seasonal_mean0[0]['precip'].to_numpy(), rtol=1e-5)
    var_mean_dcp = [((mdcp - smn) ** 2).mean('hour') for mdcp, smn in zip(mean_dcp, seasonal_mean)]
    var_daily_mean = [((dmn - smn) ** 2).mean('time') for dmn, smn in zip(daily_mean, seasonal_mean)]
    var_seasonal_mean = [((raw - smn) ** 2).mean('time') for raw, smn in zip(three_hourly_data, seasonal_mean)]
    var_residual = [var_smn - var_mdcp - var_dmn for var_smn, var_mdcp, var_dmn in zip(var_seasonal_mean, var_mean_dcp, var_daily_mean)]

    utils.region_plot(
        [np.sqrt(da['precip'].to_numpy()) for da in var_residual],
        PLOT_PARAMS | {'cmap': 'variance'},
        LABELS,
        '',
        f'./models/{model_name}/std_residual_{note}.png',
        figsize=(12, 4)
    ) 
    utils.region_plot(
        [np.sqrt(da['precip'].to_numpy()) for da in var_daily_mean],
        PLOT_PARAMS | {'cmap': 'variance'},
        LABELS,
        '',
        f'./models/{model_name}/std_daily_mean_{note}.png',
        figsize=(12, 4)
    ) 
    utils.region_plot(
        [np.sqrt(da['precip'].to_numpy()) for da in var_mean_dcp],
        PLOT_PARAMS | {'cmap': 'variance'},
        LABELS,
        '',
        f'./models/{model_name}/std_mean_dcp_{note}.png',
        figsize=(12, 4)
    ) 

# ---------------------------------------------------------------------------------
def plot_rx(data: list[xr.DataArray], model_name: str, note: str) -> None:
    is_daily = (data[0].time[1] - data[0].time[0]).astype('timedelta64[D]') == np.timedelta64(1, 'D')
    if is_daily:
        f_head = 'rx1d'
    else:
        f_head = 'rx3h'
    rx = [da.groupby('time.year').max('time').mean('year') for da in data]  # this right..?
    utils.region_plot(
        [da['precip'].to_numpy() for da in rx],
        PLOT_PARAMS | {'cmap': 'clima'},
        LABELS,
        '',
        f'./models/{model_name}/{f_head}_{note}.png',
        figsize=(12, 4)
    ) 

# ---------------------------------------------------------------------------------
def relative_daily_intensity(daily_data: list[xr.DataArray], model_name: str, note: str) -> None:
    ### scaled intensity plot
    thresh = np.linspace(0, 100, 101)
    daily_P0 = [lat_wtd_mean(da.where(da >= 0, 0), dim=['lat', 'lon']).mean(dim='time')['precip'].to_numpy() for da in daily_data]
    print(daily_P0)
    daily_Pr = []
    for t in thresh:
        out = [lat_wtd_mean(da.where(da >= t, 0), dim=['lat', 'lon']).mean(dim='time')['precip'].to_numpy() for da in daily_data]
        daily_Pr.append(out)

    daily_Pr = np.asarray(daily_Pr)
    fig, axs = plt.subplots(1, 2, layout='constrained')
    ax = axs.flatten()
    for i, l, c in zip(range(len(daily_data)), LABELS, COLORS):
        ax[0].plot(thresh, daily_Pr[:, i], label=l, color=c)
        ax[1].plot(thresh, daily_Pr[:, i] / daily_P0[i], label=l, color=c)
    ax[0].set(ylim=(0, 3.5), xlim=(0, 100), box_aspect=1, title='GP')
    ax[1].set(ylim=(0, 1), xlim=(0, 100), box_aspect=1, title='GPnorm')
    plt.legend()
    plt.savefig(f'./models/{model_name}/intensity_pdf_{note}.png', dpi=300)
    plt.clf()

# ---------------------------------------------------------------------------------
def plot_pdf(data: list[xr.DataArray], model_name: str, note: str) -> None:
    is_daily = (data[0].time[1] - data[0].time[0]).astype('timedelta64[D]') == np.timedelta64(1, 'D')
    if is_daily:
        xlabel = 'Daily Accumulated Precipitation [mm/day]'
        f_head = 'd1'
    else:
        xlabel = '3-hourly Accumulated Precipitation [mm/3h]'
        f_head = 'h3'
    flat_data = [da.to_dataarray().data.flatten() for da in data]
    mask = [~np.isnan(arr) for arr in flat_data]
    masked_data = [arr[m] for arr, m in zip(flat_data, mask)]
    df_dict = {}
    for arr, lab in zip(masked_data, LABELS):
        print(lab, len(arr))
        df_dict[lab] = arr
    df = pd.DataFrame({k: pd.Series(v) for k, v in df_dict.items()})
    df = df.melt(var_name='models', value_name='precip')
    ax = sns.histplot(data=df, **PDF_KWARGS)
    ax.set(ylabel='Probability', ylim=(10 ** -2, 10), xlabel=xlabel, xlim=(0.3, 3 * 10 ** 2))
    plt.savefig(f'./models/{model_name}/{f_head}_pdf_{note}.png', dpi=300)
    plt.clf()

# ---------------------------------------------------------------------------------
def plot_extremes(data: list[xr.DataArray], model_name: str, note: str) -> None:
    is_daily = (data[0].time[1] - data[0].time[0]).astype('timedelta64[D]') == np.timedelta64(1, 'D')
    if is_daily:
        ylabel = 'Daily Accumulated Precipitation [mm/day]'
        f_head = 'd1'
    else:
        ylabel = '3-hourly Accumulated Precipitation [mm/3h]'
        f_head = 'h3'

    ## 50-99.99 percentile
    out = []
    perc = np.concat([
        np.arange(50, 91, 5),
        np.linspace(91, 99, 17),
        np.linspace(99.1, 99.9, 17),
        np.linspace(99.91, 99.99, 17)
    ])
    for p in perc:
        out.append([np.nanpercentile(da.to_dataarray().data.flatten(), p) for da in data])

    out = np.asarray(out)
    n = 5
    x_int = 1 - 1 / 10 ** np.arange(n + 2)
    ticklabs = [f'{100 * v:.2f}%' for v in x_int]
    fig, ax = plt.subplots()
    for i, l, c in zip(range(len(data)), LABELS, COLORS):
        ax.plot(100 - perc, out[:, i], label=l, color=c)
    ax.invert_xaxis()
    ax.set(
        yscale='log',
        ylim=(0.1, 200),
        xscale='log',
        xlabel='Percentile of distribution',
        ylabel=ylabel
    )
    ax.xaxis.set_ticklabels(ticklabs[::-1])
    plt.legend()
    plt.savefig(f'./models/{model_name}/{f_head}_extremes_{note}.png', dpi=300)

# ---------------------------------------------------------------------------------
def plot_precip(complete, time_slice, model_name, note, cesm=False):
    season = model_name.split('_')[1]
    match season:
        case 'MAMJJA':
            selected = ['2017-08-04', '2008-05-12', '2007-04-18', '2015-07-26']
        case 'JJA':
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
            PLOT_PARAMS,
            model_name,
            note=note
        )

# ---------------------------------------------------------------------------------
def annual_stats(complete, model_name, note):
    # per-year daily mean
    annual = [da.resample(time='D').sum(dim='time').dropna('time', how='all').groupby('time.year').mean(dim='time') for da in complete]

    # total annual mean daily precip
    climatology = {}
    for da, _id in zip(annual, LABELS):
        climatology |= {str(yr) + _id: da.sel(year=yr)['precip'].as_numpy() for yr in da.year.values}

    utils.plot_mean(
        climatology,
        PLOT_PARAMS,
        model_name,
        note=note + '_annual',
        bias=False
    )

    # total annual mean daily precip bias
    bias = annual[0] - annual[1]
    utils.plot_mean(
        {str(yr): bias.sel(year=yr)['precip'].as_numpy() for yr in bias.year.values},
        PLOT_PARAMS,
        model_name,
        note=note + '_annual',
        bias=True
    )

# ---------------------------------------------------------------------------------
def daily_stats(complete):
    # resample to regional latitude-weighted daily means
    gp = [da.sel(lat=GP_SLAT, lon=GP_SLON) for da in complete]
    daily = [lat_wtd_mean(da.resample(time='D').sum(dim='time').dropna('time', how='all'), dim=['lat', 'lon'])['precip'] for da in gp]
    #diff = (daily[0] - daily[1]).sortby(lambda x: x, ascending=False)
    min_obs = [da.sortby(lambda x: x, ascending=False) for da in daily]
    return min_obs[1].time.values[0]

# ---------------------------------------------------------------------------------
def weekly_stats(complete):
    # resample to regional latitude-weighted daily means
    ngp = [da.sel(lat=NGP_SLAT, lon=SLON) for da in complete]
    weekly = [lat_wtd_mean(da.resample(time='7D').sum(dim='time').dropna('time', how='all'), dim=['lat', 'lon'])['precip'] for da in ngp]
    #diff = (daily[0] - daily[1]).sortby(lambda x: x, ascending=False)
    min_obs = [da.sortby(lambda x: x, ascending=False) for da in weekly]
    return min_obs[1].time.values[0]

# ---------------------------------------------------------------------------------
def plot_seasonal_stats(seasonal, model_name, note):
    season = model_name.split('_')[1]

    # total seasonal mean daily precip
    climatology = {}
    for da, _id in zip(seasonal, LABELS):
        climatology |= {_id: da.sel(season=ssn)['precip'].to_numpy() for ssn in da.season.values}

    utils.plot_mean(
        climatology,
        PLOT_PARAMS,
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
        PLOT_PARAMS,
        model_name,
        note=note + '_seasonal',
        bias=True
    )

# ---------------------------------------------------------------------------------
def spatial_phase_plot(three_hourly_data: list[xr.DataArray], model_name: str, note: str) -> None:
    # seasonal daily mean is what they are looking for
    seasonal_daily_mean = [da.resample(time='D').sum(dim='time', skipna=True).dropna('time', how='all').mean(dim='time') for da in three_hourly_data]
    hourly_mean = [da.groupby('time.hour').mean(dim='time', skipna=True).rename({'hour': 'time'}) * 8 for da in three_hourly_data]

    phase_out = []
    for da_h, da_m in zip(hourly_mean, seasonal_daily_mean):
        fft_out = xr.apply_ufunc(
            np.fft.fft,
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

    utils.region_plot(
        phase_out,
        PLOT_PARAMS | {'cmap': 'phase'},
        LABELS,
        '',
        f'./models/{model_name}/spatial_phase_{note}.png',
        figsize=(12, 4)
    )
    return

# ---------------------------------------------------------------------------------
def dial_plot(three_hourly_data: list[xr.DataArray], model_name: str, note: str) -> None:
    ## harmonic dial plot; take this spatially-explicit mean diurnal cycle and average across gridpoints
    hourly_mean = [da.groupby('time.hour').mean(dim='time', skipna=True).rename({'hour': 'time'}) * 8 for da in three_hourly_data]
    regional_hourly_mean = [lat_wtd_mean(da, dim=['lat', 'lon'])['precip'] for da in hourly_mean]
    fig, ax = plt.subplots(layout='constrained', subplot_kw={'projection': 'polar'})
    dial_out = []
    for da, l, c in zip(regional_hourly_mean, LABELS, COLORS):
        fft_out = xr.apply_ufunc(
            np.fft.fft,
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
        # NOTE(ereastin): I think this isn' quite right
        diu = fft_out.isel(frequency=1)
        amp, phs = np.abs(diu), xr.ufuncs.angle(diu)
        ax.plot(phs, amp, 'o', color=c, label=l)
        ax.plot([0, phs], [0, amp], '-', color=c)

    ax.grid(True)
    ax.legend()
    plt.savefig(f'./models/{model_name}/dial_{note}.png', dpi=300)
    return

# ---------------------------------------------------------------------------------
def diurnal_line_plot(three_hourly: list[xr.Dataset], model_name: str, note: str) -> None:
    # regional mean diurnal cycle line plot
    regional_mean = [lat_wtd_mean(da, dim=['lat', 'lon'])['precip'] for da in three_hourly]
    # TODO: theres a way to not hard code this but bit of a pain
    time_regional_mean = [da.groupby('time.hour').mean(dim='time', skipna=True) * 8 for da in regional_mean]
    fig, ax = plt.subplots(layout='constrained')
    for da, _id, c in zip(time_regional_mean, LABELS, COLORS): da.plot.line(ax=ax, color=c, label=_id)
    ax.set(xlabel='UTC Hour', ylabel='Precipitation [mm/d]', xlim=(0, 21), xticks=np.arange(0, 22, 3))
    ax.legend(frameon=False, fancybox=False, fontsize='small')
    plt.savefig(os.path.join(pth.MODEL_OUT, f'{model_name}', f'diurnal_line_{note}.png'), dpi=300)

# ---------------------------------------------------------------------------------
def hovmoller_plot(three_hourly: list[xr.Dataset], model_name: str, note: str) -> None:
    # Hovmoller diagrams
    regional_mean = [lat_wtd_mean(da, dim=['lat'])['precip'] for da in three_hourly]
    time_regional_mean = [da.groupby('time.hour').mean(dim='time', skipna=True) * 8 for da in regional_mean]
    N = len(time_regional_mean)
    fig, axs = plt.subplots(1, N, figsize=(12, 6), sharey=True, layout='constrained')
    axs = axs.flatten()
    cmap = cm.YlGnBu
    vmin, vmax = 0, 6
    extend = 'max'
    ## for diffs
    #cmap = cm.BrBG
    #vmin, vmax = -3, 3
    #extend = 'both'
    for da, ax, _id in zip(time_regional_mean, axs, LABELS):
        a = da.plot.contourf(
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            levels=13,
            extend=extend,
            cmap=cmap,
            add_colorbar=False
        )
        ax.set(
            title=f'{_id}',
            box_aspect=1,
            yticks=np.arange(0, 24, 3),
            xticks=np.arange(-105, -80, 5),
            ylabel='',
            xlabel='Longitude [deg E]'
        )
    axs[0].set(ylabel='UTC Hour')
    fig.colorbar(a, ax=axs, location='right', orientation='vertical', label='[mm/d]', shrink=0.4)
    plt.savefig(os.path.join(pth.MODEL_OUT, f'{model_name}', f'hovmoller_{note}.png'), dpi=300)

# ---------------------------------------------------------------------------------
## O'Gorman scaling stuff
def process_scaling(model_name, exp, daily=True, cesm=False):
    ## CESM fields
    ## um big problem here.. cant calc this because of nan filling...
    lsf_path = os.path.join(pth.SCRATCH, 'cus_cesm', f'LSF*.{exp}.nc')
    lsf = open_mf(lsf_path, drop_vars=['U', 'V', 'Z3'], lat=CUS_LAT, lon=CUS_LON)
    lsf = lsf.sel(time=lsf.time.dt.month.isin([6, 7, 8]))

    ## Surface pressure from CESM - needed for scaling calcs
    ps_path = os.path.join(pth.SCRATCH, 'cus_cesm', f'PS*.{exp}.nc')
    ps = open_mf(ps_path, lat=CUS_LAT, lon=CUS_LON)
    ps = ps.sel(time=ps.time.dt.month.isin([6, 7, 8]))
    
    ## TODO: how about scaling for MERRA
    #lsf_path = os.path.join(pth.MERRA, 'M2I3NPASM.1980')
    #lsf = open_mf()

    ## Surface precip - DL or CESM
    cesm_path = f'./CESM.{exp}_1979-1983.nc'
    dl_path = f'./models/{model_name}/MODEL_PRED_CESM.{exp}.nc'
    pr_path = cesm_path if cesm else dl_path
    pr = xr.open_dataset(pr_path)
    if not cesm: pr = pr.rename({'precip': 'PRECT'})
    pr = pr.sel(time=pr.time.dt.month.isin([6, 7, 8]))

    if daily:  # comfortable with results of this.?
        lsf = lsf.resample(time='D').mean(dim='time').dropna('time', how='all')
        Rx1day = pr.resample(time='D').sum(dim='time').dropna('time', how='all')
        ps = ps.resample(time='D').mean(dim='time').dropna('time', how='all')

    ## expand dims across pressure levels for indexing
    pr_tall = pr.expand_dims(dim={'plev': lsf.plev}).transpose('time', 'plev', ...)
    ps_tall = ps.expand_dims(dim={'plev': lsf.plev}).transpose('time', 'plev', ...)

    comb = xr.merge([lsf, pr_tall, ps_tall])

    yr_out = []
    for yr in range(1979, 1984):
        a = comb.sel(time=str(yr))['PRECT'].idxmax(dim='time')
        yr_out.append(comb.sel(time=a))

    yr_dim = xr.DataArray(list(range(1979, 1984)), dims='year', name='year')
    max_all = xr.concat(yr_out, dim=yr_dim)
    out_f = f'./CESM.{exp}_scaling' if cesm else f'./models/{model_name}/CESM.{exp}_DL_scaling'
    out_f += '_D1' if daily else '_h3'
    max_all.to_netcdf(out_f + '.nc', engine='netcdf4')

    ## Run scaling calc for surface precip
    scaling = xr.open_dataset(out_f + '.nc')
    scaling['plev'] = scaling['plev'] * 100
    scaling['plev'].attrs['units'] = 'Pa'
    scaling['QS'] = calc_qsat(scaling['T'], scaling['plev'])
    scaling['diffQS'] = scaling['QS'].differentiate('plev')  # kg_v/(kg_a*Pa)
    scaling = scaling.drop_vars('time')
    scaling = scaling.rename({'plev': 'lev'})
    g = -9.80665
    # NOTE: this is crazy slow.. can just do by hand.?
    weight = gc.meteorology.delta_pressure(pressure_lev=scaling.lev, surface_pressure=scaling.PS.isel(lev=0)) / g
    # filter for regions of ascent and below 50 hPa w/lapse rate of 2 K/km, just using 50 hPa thresh rn
    filtered = scaling.sel(lev=slice(100000, 5000))[['OMEGA', 'diffQS']].where(scaling.OMEGA < 0)
    scaling_P = filtered['OMEGA'] * filtered['diffQS'] * weight  # kg_v/(m2 * s)
    n_sec = 3600 * 24 if daily else 3600 * 3
    out = scaling_P.sum(dim='lev', skipna=True) * n_sec
    out_f += '_P'
    out.to_netcdf(out_f + '.nc', engine='netcdf4')
    return

# ---------------------------------------------------------------------------------
def plot_scaling(model_name, exp, daily=True):
    note = 'idk'
    ### Model surface precip
    ## MSWEPv2 Rx1day
    mswep_pr = xr.open_dataset(f'./MSWEP_1979-1983.nc')

    ## CESM Rx1day
    cesm_pr = xr.open_dataset(f'./CESM.{exp}_1979-1983.nc')
    cesm_pr = cesm_pr.rename({'PRECT': 'precip'})

    ## DL model Rx1day
    dl_pr = xr.open_dataset(f'./models/{model_name}/MODEL_PRED_CESM.{exp}.nc')

    ## Resample, select for JJA, get mean annual max
    model_pr = [mswep_pr, dl_pr, cesm_pr]
    #model_pr = [dl_pr, cesm_pr]
    #model_pr = [cesm_pr]
    if daily:
        model_pr = [da.resample(time='D').sum(dim='time').dropna('time', how='all') for da in model_pr]
    
    model_pr = [da.sel(time=da.time.dt.month.isin([6, 7, 8])) for da in model_pr]
    model_pr = [da.groupby('time.year').max(dim='time').mean(dim='year') for da in model_pr]

    ### Scaling-based precip
    ## CESM Rx1day from scaling argument
    tag = 'D1' if daily else 'h3'
    cesm_scaling_pr = xr.open_dataset(f'./CESM.{exp}_scaling_{tag}_P.nc') / 8   # NOTE: div8 bc of screwup in processing

    ## DL model Rx1day from scaling argument
    dl_scaling_pr = xr.open_dataset(f'./models/{model_name}/CESM.{exp}_DL_scaling_{tag}_P.nc')

    model_scaling_pr = [dl_scaling_pr, cesm_scaling_pr]
    #model_scaling_pr = [cesm_scaling_pr]
    model_scaling_pr = [da.mean(dim='year').rename({'__xarray_dataarray_variable__': 'precip'}) for da in model_scaling_pr]

    ## Do plotting
    #datas = [dls_data.to_numpy() / dlr_data.to_numpy()]
    #titles = ['DL Scaling : DL Rx1day']
    #utils.region_plot(datas, PLOT_PARAMS | {'cmap': 'ratio'}, titles, 'a', './dl_noqw_ratio.png', figsize=(12, 4))
    data = {l + ' Rx3h': da['precip'].to_numpy() for l, da in zip(LABELS, model_pr)}
    data |= {l + ' Sc': da['precip'].to_numpy() for l, da in zip(LABELS[1:], model_scaling_pr)}
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
def animate(model_name, plot_params, time, cesm=True, exp=''):
    tsel = time.start[:6].replace('-', '_')
    kwargs = {'lat': CUS_LAT, 'lon': CUS_LON, 'time': time}
    pr_kwargs = {'drop_vars': [], 'time': time}
    if cesm:
        lsf_fpath = os.path.join(pth.SCRATCH, 'cus_cesm', f'LSF_{tsel}*.{exp}.nc')
        obs_p_fpath = os.path.join('CESM.{exp}_1978-1983.nc'),
        dl_p_fpath = os.path.join('models', model_name, 'MODEL_PRED_CESM.{exp}.nc')
        kwargs |= {'drop_vars': ['T', 'U', 'V', 'Z2']}
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
    fig, (ax0, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 4), dpi=350, subplot_kw={'projection': map_proj})
    axs = [ax0, ax2, ax3, ax4]
    for ax in axs: ax.add_feature(cfeat.STATES, linewidth=0.25)
    ax0.set_title('DL Surface P', fontsize='small')
    if cesm:
        ax1.set_title('CESM Surface P', fontsize='small')
        ax2.set_title(r'CESM $\omega$ 550hPa', fontsize='small')
        ax3.set_title(r'CESM TCWV', fontsize='small')
    else:
        ax1.set_title('MSWEPv2 Surface P', fontsize='small')
        ax2.set_title(r'MERRA2 $\omega$ 550hPa', fontsize='small')
        ax3.set_title(r'MERRA2 TCWV', fontsize='small')

    cbar_ticks = [np.arange(0, 17, 4), np.arange(0, 17, 4), np.linspace(-1, 1, 5), np.arange(0, 61, 10)]
    cbar_labels = ['[mm]', '[mm]', '[Pa/s]', r'[kg/m$^1$]']
    #area = surface_area(lsf)
    lsf['lev'] = lsf['lev'] * 100 ## * 99.?!
    lsf['lev'].attrs['units'] = 'Pa'
    for i, dt in enumerate(dl_p.time):
        # TODO: this correct?
        cq = lsf.sel(time=dt)['QV'].integrate('lev') / -9.8  # tf is 10.8..
        co = lsf.sel(time=dt, lev=550 * 100)['OMEGA'] ## 549..?
        cp, dp = obs_p.sel(time=dt)['precip'], dl_p.sel(time=dt)['precip']
        items = utils.region_ani([dp, cp, co, cq], plot_params, ax0, ax2, ax3, ax4)
        title = plt.text(0.1, 1.3, f'TIME: {dt.values} UTC', horizontalalignment='center', verticalalignment='bottom', transform=ax2.transAxes)
        items.insert(-1, title)
        artists.append(items)
        if i == -1:
            for j, (item, ax, t, l) in enumerate(zip(items[0:], axs, cbar_ticks, cbar_labels)):
                fig.colorbar(item, ax=ax, location='bottom', ticks=t, label=l)

    ani = animation.ArtistAnimation(fig, artists, interval=500)
    ani.save(filename=f'./models/{model_name}/ani.mp4', writer='ffmpeg')
    return

# ---------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
