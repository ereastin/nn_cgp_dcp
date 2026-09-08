import pandas as pd
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MERRA_VARS = ['QV', 'U', 'V', 'OMEGA', 'H', 'T']
SHRT_MERRA_VARS = ['U', 'V', 'H', 'T']

LEV = np.array([
    1000, 975, 950, 925, 900, 875, 850,
    825, 775, 700, 600, 550, 450, 400, 350, 300,
    250, 200, 150, 100, 70, 
    50, 40, 30, 20, 10, 7, 3
])

## ================================================================================
def run():
    fig, axs = plt.subplots(1, 2, layout='constrained', sharey=True)
    metric = 'ets'
    models = ['test2_JJA_F00h', 'noQW_JJA_F00h']
    for model_name, ax in zip(models, axs):
        stats = []
        for v in MERRA_VARS:
            try:
                df = pd.read_csv(f'./models/{model_name}/{v}_shuffle.csv', index_col=0)
            except FileNotFoundError as e:
                continue
            for index, data in df.iterrows():
                stats.append({
                    'label': index,
                    'mean': data[f'{metric}_imp_mean'],
                    'med': data[f'{metric}_imp_med'],
                    'q1': data[f'{metric}_imp_25p'],
                    'q3': data[f'{metric}_imp_75p'],
                    'whislo': data[f'{metric}_imp_25p'],
                    'whishi': data[f'{metric}_imp_75p']
                })
        stats = sorted(stats, key=lambda x: x['med'], reverse=True)
        stats = stats[:10]
        ax.bxp(stats, showfliers=False, showmeans=True, meanline=True)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, rotation=45, ha='right', rotation_mode='anchor')
        ax.axhline(y=0, color='gray', linewidth=1, linestyle='--')

    axs[0].set(title='DL All', ylabel=f'{metric.upper()} Importance', box_aspect=1)
    axs[1].set(title='DL Large-scale', box_aspect=1)
    return fig
    #plt.savefig(f'./models/combined_{metric}_bxplt.png', dpi=300)

## ================================================================================
def combine(model_name):
    v = 'OMEGA'
    df_add = pd.read_csv(f'./models/{model_name}/part_{v}_shuffle.csv', index_col=0)
    df = pd.read_csv(f'./models/{model_name}/{v}_shuffle.csv', index_col=0)
    out = pd.concat([df_add, df])
    out.to_csv(f'./models/{model_name}/cum_{v}_shuffle.csv')
    return

## ================================================================================
def log_to_csv(model_name):
    v = 'OM'
    with open(f'./_logs/{v}shuf_test2_1779191.out', 'r') as f:
        content = f.read()
        chunks = content.split('*' * 60)[1:-1]

    df_list = []
    for c in chunks:
        print(c)
        lines = c.split('\n')
        pert_dict_st = lines[1].find('{')
        a = eval(lines[1][pert_dict_st:])
        print(a)
        df_idx = a[0]['var'] + str(a[0]['levels'])
        #mean_mse = float(lines[2].split(':')[-1].strip())
        #mean_pcc = float(lines[3].split(':')[-1].strip())
        #n_good_pccs = int(lines[4].split(':')[-1].split('of')[0].strip())
        #mean_ets = float(lines[5].split(':')[-1].strip())
        #bias = float(lines[6].split(':')[-1].strip())
        #rmse_met = lines[8].split()
        [rmn, rmd, r25, r75] = [float(s.strip()) for s in lines[22].split()]
        #ets_met = lines[9].split()
        [emn, emd, e25, e75] = [float(s.strip()) for s in lines[23].split()]
        df = pd.DataFrame({
            #'mean_mse': mean_mse,
            #'mean_pcc': mean_pcc,
            #'n_good_pcc': n_good_pccs,
            #'mean_ets': mean_ets,
            'rmse_imp_mean': rmn,
            'rmse_imp_med': rmd,
            'rmse_imp_25p': r25,
            'rmse_imp_75p': r75,
            'ets_imp_mean': emn,
            'ets_imp_med': emd,
            'ets_imp_25p': e25,
            'ets_imp_75p': e75,
            #'mean_reg_bias': bias
        }, index=[df_idx])
        df_list.append(df)

    out = pd.concat(df_list)
    print(out)
    out.to_csv(f'./models/{model_name}/part_{v}_shuffle.csv')

## ================================================================================
if __name__ == '__main__':
    main()
