import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
import os
import json

def main():
    N = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    cluster = LocalCluster(n_workers=N, memory_limit=None)
    with Client(cluster) as client:
        print(cluster)
        print(client)
        pth = '/scratch/eastinev/cus_cesm/'
        cesm_files = [f for f in os.listdir(pth) if 'LSF' in f]
        ctrl_files = [os.path.join(pth, f) for f in cesm_files if 'CTRL' in f]
        warming_files = [os.path.join(pth, f) for f in cesm_files if '2K' in f]
        variables = ['Q', 'U', 'V', 'T', 'OMEGA', 'Z3']
        
        ctrl_ds = xr.open_mfdataset(
            ctrl_files,
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
        ctrl_stats = {k: {'mn': None, 'std': None} for k in variables}
        for v in variables:
            print(v)
            ctrl_stats[v]['mn'] = float(ctrl_ds[v].mean().data.compute())
            ctrl_stats[v]['std'] = float(np.sqrt(ctrl_ds[v].var().data.compute()))
        with open('./CESM.CTRL_stats.json', 'w') as f:
            json.dump(ctrl_stats, f)
        ctrl_ds.close()

        warming_ds = xr.open_mfdataset(
            warming_files,
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
        warming_stats = {k: {'mn': None, 'std': None} for k in variables}
        for v in variables:
            print(v)
            warming_stats[v]['mn'] = float(warming_ds[v].mean().data.compute())
            warming_stats[v]['std'] = float(np.sqrt(warming_ds[v].var().data.compute()))

        with open('./CESM.2K_stats.json', 'w') as f:
            json.dump(warming_stats, f)
        warming_ds.close()

if __name__ == '__main__':
    main()
