import os
import json
import numpy as np
import pandas as pd
import math
import itertools
from scipy.stats import kendalltau

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig
import seaborn as sns
matplotlib.use('Agg')
sns.set_style("whitegrid")
plt.rcParams["font.family"] = 'DejaVu Sans'

from plot_utils import *

save_dir = 'graphs-combine-subtasks'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

num_source_models = 300
benchmarks = ['mmlu', 'mmlu-pro', 'bbh', 'gpqa']
techniques = ['Random (uniform)', 'Random (subtask stratified, equal)', 'Stratified sampling (confidence)', 'Diversity', 'Anchor Points', 'tinyBenchmarks']
all_fraction_sampled_points = [0.02, 0.04, 0.08, 0.16, 0.24, 0.32, 0.4]

f, axs = plt.subplots(len(benchmarks), len(techniques), figsize=(6.4*len(techniques), 3.6*len(benchmarks)))#, gridspec_kw={'height_ratios': [0.5, 0.5, 1]})
plt.subplots_adjust(wspace=0.25, hspace=0.25)    

for ax_row, benchmark in enumerate(benchmarks):

    cached_mde_results_fn = f'{save_dir}/{benchmark}-mde-results.csv'
    cached_estimation_results_fn = f'{save_dir}/{benchmark}-estimation-results.csv'

    if not os.path.exists(cached_mde_results_fn) and not os.path.exists(cached_estimation_results_fn):
        print(f'Processed results files not generated for {benchmark}')
        continue
        
    df_mde = pd.read_csv(cached_mde_results_fn)
    full_estimation_df = pd.read_csv(cached_estimation_results_fn)

    # get difference between seen and unseen
    config_to_seen = {}
    config_to_unseen = {}
    for i, row in df_mde.iterrows():
        config = (row['Technique'], row['Number of source models'], row['Fraction of sampled points'])
        m = row['MDE']
        l = row['MDE lower CI']
        u = row['MDE upper CI']
        y1 = min(l, m-(u-m))
        y2 = max(m+(m-l), u)
        if row['Split'] == 'seen':
            config_to_seen[config] = (m, y1, y2)
        else:
            config_to_unseen[config] = (m, y1, y2)
    tidy_mde_results = []
    for config in config_to_seen.keys():
        tidy_mde_results.append({'Technique': config[0],
                                 'Number of source models': config[1],
                                 'Fraction of sampled points': config[2],
                                 'MDE (seen split)': config_to_seen[config][0],
                                 'MDE lower CI (seen split)': config_to_seen[config][1],
                                 'MDE upper CI (seen split)': config_to_seen[config][2],
                                 'MDE (unseen split)': config_to_unseen[config][0],
                                 'MDE lower CI (unseen split)': config_to_unseen[config][1],
                                 'MDE upper CI (unseen split)': config_to_unseen[config][2],
                                 'MDE difference': config_to_unseen[config][0] - config_to_seen[config][0]})
    df_mde = pd.DataFrame(tidy_mde_results)

    all_fraction_sampled_points = [10, 25, 50, 100, 250, 500, 1000]
    if benchmark == 'gpqa':
        all_fraction_sampled_points = [10, 25, 50, 100, 200]
    
    xticklabels = [f'{k}\n({k/benchmark_to_full_seen_size[benchmark]*100:.1f}%)' for k in all_fraction_sampled_points]
    
    for ax_col, technique in enumerate(techniques):
        ax = axs[ax_row, ax_col]
        plot_df = df_mde.loc[(df_mde['Number of source models'] == num_source_models) \
                        & (df_mde['Technique'] == technique)]
        sns.lineplot(ax=ax, data=plot_df, x='Fraction of sampled points', y=f"MDE (seen split)", hue="Technique", palette=palette, marker='o', markersize=10, alpha=0.8, linewidth=4, zorder=10, clip_on=True)
        for split in ['seen', 'unseen']:
            for t in techniques:
                technique_df = plot_df.loc[(plot_df['Technique'] == t)]
                data = sorted(zip(technique_df['Fraction of sampled points'].tolist(), technique_df[f'MDE lower CI ({split} split)'].tolist(),  technique_df[f'MDE upper CI ({split} split)'].tolist()))
                xs = [x for x, _, _, in data]
                y1s = [y1 for _, y1, _ in data]
                y2s = [y2 for _, _, y2 in data]
                ax.fill_between(xs, y1s, y2s, color=palette[technique], alpha=0.2)
        sns.lineplot(ax=ax, data=plot_df, x='Fraction of sampled points', y=f"MDE (unseen split)", hue="Technique", palette=palette, linestyle='--', marker='s', markersize=10, alpha=0.8, linewidth=4, zorder=10, clip_on=True)
        ax.set_xscale('log')
        ax.set_ylim((0, 0.3))
        ax.set_xticks(all_fraction_sampled_points)
        ax.set_xticklabels(xticklabels)
        yticks = ax.get_yticks()
        yticklabels = [f'{y * 100:.0f}' for y in yticks]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)    
        ax.set_xlabel('')
        ax.set_ylabel('')
        if ax_col == 0:
            ax.set_ylabel(f'MDAD', fontsize=14, labelpad=20, fontweight='bold')
            ax.text(-0.1, 0.5, '↓', fontsize=24, fontweight='bold', va='center', ha='center', transform=ax.transAxes)
            ax.text(-0.25, 0.5, benchmark_to_display_name[benchmark], transform=ax.transAxes, rotation_mode='anchor', va='center', ha='center', fontsize=24, fontweight='bold', rotation=90)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        if ax_row == 0:
            ax.set_title(technique, fontsize=18, fontweight='bold', pad=10)
        if ax_row == len(benchmarks) - 1:
            ax.set_xlabel('Number of examples selected for micro-benchmark', fontsize=10, fontweight='bold')
            
ls = [matplotlib.patches.Patch(facecolor=palette[t]) for t in techniques]
line1 = matplotlib.lines.Line2D([], [], color='black', linewidth=4,)
line2 = matplotlib.lines.Line2D([], [], color='black', linewidth=4, linestyle='--')
plt.legend([line1, line2], ['Micro-benchmark on full dataset used to construct it', 'Micro-benchmark on new draw of the task'], bbox_to_anchor=(0.5, 0.05), loc='upper center', bbox_transform=f.transFigure, borderaxespad=0., fontsize=20, ncols=2)#, prop={'weight':'bold'})
savefig(f'{save_dir}/figure-6_combine-subtasks_all-benchmarks.pdf', bbox_inches='tight')