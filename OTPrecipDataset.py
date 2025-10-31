import torch
from torch.utils.data import Dataset
import dask
from dask.distributed import Client, LocalCluster
import xarray as xr
import numpy as np
import cftime
import os
import sys
sys.path.append('/home/eastinev/ai')
import paths as pth
import time
import json
from itertools import product
import mcs_data as mcs
import file_utils as futils

## ================================================================================
STATS = False
DRY = False  # TODO: allow assignment from outside

## ================================================================================
def main():
    n_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    cluster = LocalCluster(n_workers=n_cpus, memory_limit=None)  # have to specify this think it defaults to 4 or smthn
    print(cluster, flush=True)
    with Client(cluster) as client:
        print(client, flush=True)
        pd = OTPrecipDataset('train', 'cus', 'nR1_JJA_F00h', standardize=False, sel_mcs=True)
        t1 = time.time()
        for i, (s, t, tt) in enumerate(pd):
            print(tt)

    if STATS: pd.get_stats()
    t2 = time.time()
    print('time:', (t2 - t1))

## ================================================================================
class OTPrecipDataset(Dataset):
    def __init__(
        self,
        mode,
        exp,
        model_name,
        standardize=True,
        shuffle=False,
        ret_as_tnsr=True,
        sel_mcs=True,
        drop_vars=[],
        cesm_exp=''
    ):
        super(OTPrecipDataset, self).__init__()
        self.mode = mode  # train, val, test, all, cesm
        self.exp = exp  # rename? identifies cus dir rn
        self.model_name = model_name
        self._STANDARDIZE = standardize
        self._SHUFFLE = shuffle
        self._RET_AS_TNSR = ret_as_tnsr
        self._MCS = sel_mcs
        self._CESM = True if mode == 'cesm' else False
        self._CESM_EXP = cesm_exp
        self._drop_vars = drop_vars

        _, self.season, forecast = self.model_name.split('_')
        self._LEAD_TIME_HOURS = int(forecast[1:3])
        self._FORECAST = False if self._LEAD_TIME_HOURS == 0 else True
        n_months = len(self.season)
        mn_offset = 'JFMAMJJASOND'.find(self.season) + 1

        yrs = list(range(2004, 2021)) if not self._CESM else list(range(1979, 1984))
        mnths = list(range(mn_offset, mn_offset + n_months))
        wks = list(range(4))

        data_splits_path = f'./models/{self.model_name}/data_splits.json'
        if not os.path.exists(data_splits_path):
            print(f'writing data splits {data_splits_path}')
            futils.write_t_strs(yrs, mnths, wks, self.model_name)
        with open(data_splits_path, 'r') as f:
            print(f'loading data splits for mode {mode}: {data_splits_path}')
            data_splits = json.load(f)

        match mode:
            case 'train':
                self.t_strs = data_splits['train']
            case 'val':
                self.t_strs = data_splits['val']
            case 'test':
                self.t_strs = data_splits['test']
                # NOTE: for using ONLY dry days, set MCS, DRY = True, for all days including dry set MCS = False, DRY = True.?
                if DRY: self.t_strs += data_splits['dry']
            case 'cesm':
                self.t_strs = data_splits['cesm']
            case 'check':
                self.t_strs = data_splits['all']
            case _:
                print('Mode {mode} is not correct')
                sys.exit(21)

        ## For variable statistics/data norm
        self.merra_vars = ['U', 'V', 'OMEGA', 'H', 'T', 'QV']
        self._stats = {k + 'mn': [] for k in self.merra_vars} | {k + 'var': [] for k in self.merra_vars}
        self._stats['Nv'] = []
        self._stats['precipitationmn'] = []
        self._stats['precipitationvar'] = []
        self._stats['Np'] = []

        self.stats_pth = f'./models/{self.model_name}/norm_vars.json'
        if os.path.exists(self.stats_pth):
            print(f'Loading training set statistics from {self.stats_pth}')
            with open(self.stats_pth, 'r') as f:
                self.stats = json.load(f)

        print(f'''Dataset status :
              MODEL NAME: {self.model_name}
              EXPERIMENT: {self.exp}
              MODE: {mode}
              FORECAST: {self._FORECAST}
              LEAD TIME (HRS): {self._LEAD_TIME_HOURS}
              MCS: {self._MCS}
              DRY: {DRY}
              CESM: {self._CESM}
              CESM_EXP: {self._CESM_EXP}
              STANDARDIZE: {self._STANDARDIZE}
              SHUFFLE: {self._SHUFFLE}
              RETURN TENSOR: {self._RET_AS_TNSR}
              DROPPED VARS: {self._drop_vars}
        ''')

    # -----------------------------------------------------------------------------
    def __len__(self):
        return len(self.t_strs)

    # -----------------------------------------------------------------------------
    def __getitem__(self, idx):
        (year, month, week) = self.t_strs[idx] 
        source_rw_pth, target_rw_pth = futils._get_filepaths(
            year,
            month,
            week,
            self.exp,
            forecast=self._FORECAST,
            cesm_exp=self._CESM_EXP
        )

        # now datetime arange
        if self._CESM:
            time_id = futils._get_wk_days_no_leap(year, month, week)
        else:
            time_id = futils._get_wk_days_leap(year, month, week)

        # Filter for MCS-present timestamps
        if self._MCS:
            t = mcs.get_times(time_id)  # this returns empty list if no MCS present:
            # if ds.sel(time=[]) this selects NONE, if ds.drop_sel(time=[]) this drops NONE
        else:
            t = np.arange(time_id[0], time_id[-1] + np.timedelta64(1, 'D'), np.timedelta64(3, 'h'))
        t_shift = t + np.timedelta64(self._LEAD_TIME_HOURS, 'h') # handles forecast lead times

        if self._CESM:  # need to convert to cftime objects for indexing
            t = [cftime.DatetimeNoLeap(dt.year, dt.month, dt.day, dt.hour) for dt in t.astype(object)]
            t_shift = [cftime.DatetimeNoLeap(dt.year, dt.month, dt.day, dt.hour) for dt in t_shift.astype(object)]

        try:
            # read in source and target datasets
            source = self.read_source(
                source_rw_pth,
                sel_time=t
            )
            target = self.read_target(
                target_rw_pth,
                sel_time=t_shift
            )
        except FileNotFoundError:
            return None, None, time_id
        except OSError:
            print(source_rw_pth, target_rw_pth)
            exit()

        if STATS:
            return None, None, time_id

        if self._RET_AS_TNSR:
            source, target = self._ret_tensor(source, target)

        sample = (source, target, time_id)
        return sample

    # -----------------------------------------------------------------------------
    def _ret_tensor(self, source_ds, target_ds):
        source = torch.tensor(source_ds.to_dataarray().to_numpy())
        source_ds.close()
        source = source.permute(1, 0, 2, 3, 4)  # return as (time, var, lev, lat, lon)

        target = torch.tensor(target_ds.to_dataarray().to_numpy())
        target_ds.close()
        target = target.permute(1, 0, 2, 3)  # return as (time, 1[precip], lat, lon)

        # Shuffle within batch
        if self._SHUFFLE:
            shuff_idx = torch.randperm(source.shape[0])
            source = source[shuff_idx]
            target = target[shuff_idx]

        return source, target

    # -----------------------------------------------------------------------------
    @staticmethod
    def prune(source, target, f_id):
        # if batch size doesn't match then trim to smaller
        print(f'pruning {f_id}...')
        trim = min(source.shape[0], target.shape[0])
        return source[:trim], target[:trim]
 
    # -----------------------------------------------------------------------------
    def get_stats(self):
        print('writing var stats...')
        with open(self.stats_pth, 'w') as f:
            Nv = np.asarray(self._stats.pop('Nv'))
            Np = np.asarray(self._stats.pop('Np'))
            stats_dump = {}
            for v in self.merra_vars:
                mn, vr = np.asarray(self._stats[v + 'mn']), np.asarray(self._stats[v + 'var'])
                comb_mn = np.sum(Nv * mn) / np.sum(Nv)
                stats_dump[v + 'mn'] = comb_mn
                qc = np.sum((Nv - 1) * vr + Nv * mn ** 2)
                stats_dump[v + 'std'] = np.sqrt((qc - np.sum(Nv) * comb_mn ** 2) / (np.sum(Nv) - 1))

            mn, vr = np.asarray(self._stats['precipitationmn']), np.asarray(self._stats['precipitationvar'])
            comb_mn = np.sum(Np * mn) / np.sum(Np)
            stats_dump['precipitationmn'] = comb_mn
            qc = np.sum((Np - 1) * vr + Np * mn ** 2)
            stats_dump['precipitationstd'] = np.sqrt((qc - np.sum(Np) * comb_mn ** 2) / (np.sum(Np) - 1))

            # copy these over for using CESM data
            stats_dump['Z3mn'] = stats_dump['Hmn']
            stats_dump['Z3std'] = stats_dump['Hstd']
            stats_dump['Qmn'] = stats_dump['QVmn']
            stats_dump['Qstd'] = stats_dump['QVstd']
            stats_dump['PRECTmn'] = stats_dump['precipitationmn']
            stats_dump['PRECTstd'] = stats_dump['precipitationstd']

            json.dump(stats_dump, f)

    # -----------------------------------------------------------------------------
    def read_source(self, in_file, sel_time):
        ds = xr.open_mfdataset(in_file, drop_variables=self._drop_vars)
        ds = ds.drop_sel(time=sel_time) if DRY and self._MCS else ds.sel(time=sel_time)

        if STATS:
            for v in self.merra_vars:
                dav = ds[v]
                self._stats[v + 'mn'].append(dav.mean().data)
                self._stats[v + 'var'].append(dav.var().data)
            self._stats['Nv'].append(dav.count().data)
            return

        if self._STANDARDIZE:
            for v in ds.variables:
                if v in ['time', 'lat', 'lon', 'plev', 'lev'] + self._drop_vars:
                    continue
                ds[v] = (ds[v] - self.stats[v + 'mn']) / self.stats[v + 'std']

        return ds

    # -----------------------------------------------------------------------------
    def read_target(self, in_file, sel_time):
        ds = xr.open_mfdataset(in_file)
        ds = ds.drop_sel(time=sel_time) if DRY and self._MCS else ds.sel(time=sel_time)
        ds = ds.where(ds >= 1, 0)  # filter for i guess 3-hour accum less than 1 mm.. as estimate of daily?
 
        if STATS:
            v = 'precipitation'
            dav = ds[v]
            self._stats[v + 'mn'].append(dav.mean().data)
            self._stats[v + 'var'].append(dav.var().data)
            self._stats['Np'].append(dav.count().data)
            return
        
        if self._STANDARDIZE:
            for v in ds.variables:
                if v in ['time', 'lat', 'lon', 'plev', 'lev']:
                    continue
                ds[v] = (ds[v] - self.stats[v + 'mn']) / self.stats[v + 'std']
 
        return ds

## ================================================================================
if __name__ == '__main__':
    main()
