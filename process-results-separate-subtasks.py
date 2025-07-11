import os
import json
import numpy as np
import pandas as pd
import math
import itertools
import shutil
from scipy.stats import kendalltau
from tqdm import tqdm
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig
import seaborn as sns
matplotlib.use('Agg')
sns.set_style("whitegrid")
plt.rcParams["font.family"] = 'DejaVu Sans'

from plot_utils import *

save_dir = 'graphs-separate-subtasks'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

results_dir = './results-cached/results-separate-subtasks'


benchmarks = ['mmlu', 'mmlu-pro', 'bbh', 'gpqa']
#all_num_source_models = [10, 50, 100, 150, 200, 250, 300]
all_num_source_models = [300]

redo = False

techniques = ["Random", 'Random_Subtask_Stratified_Equal', 'DPP', 'Stratified_Random_Sampling', "Anchor_Points_Weighted", "tinyBenchmarks"]
for benchmark in benchmarks:
    all_fraction_sampled_points = [0.02, 0.04, 0.08, 0.16, 0.24, 0.32, 0.4]
    cached_mde_results_fn = f'{save_dir}/{benchmark}-mde-results.csv'
    cached_correctness_results_fn = f'{save_dir}/{benchmark}-correctness-results.csv'
    cached_estimation_results_fn = f'{save_dir}/{benchmark}-estimation-results.csv'

    if not redo and os.path.exists(cached_mde_results_fn) \
      and os.path.exists(cached_estimation_results_fn) \
      and os.path.exists(cached_mde_results_fn):
        df_mde = pd.read_csv(cached_mde_results_fn)
        full_estimation_df = pd.read_csv(cached_estimation_results_fn)
        df_correctness = pd.read_csv(cached_correctness_results_fn)
    else:
        all_tidy_results_estimation = []
        tidy_data_mde = []
        tidy_data_correctness = []
        for num_source_models in all_num_source_models:
            print(f'{benchmark}, {num_source_models} source models, processing results:')
            fn = f'{results_dir}/results_{benchmark}_{num_source_models}-source-models.json'
            with open(fn, 'r') as f:
                results = json.load(f)
            tidy_results = []
            for result in results:
                tidy_results.extend(make_tidy_results(*result))
                all_tidy_results_estimation.append(make_tidy_results_estimation(*result))            
            full_df = pd.DataFrame(tidy_results)
            datasets = sorted(list(set(full_df['Dataset'])))
            for fraction_sampled_points in tqdm(all_fraction_sampled_points):
                fraction_sampled_points_df = full_df.loc[(full_df['Number of source models'] == num_source_models) \
                            & (full_df['Fraction of sampled points'] == fraction_sampled_points)] 
                for technique in techniques:
                    df = fraction_sampled_points_df.loc[(fraction_sampled_points_df['Technique'] == technique)]    
                    if len(df) == 0:
                        continue
                    tidy_data_correctness.append({'Technique': technique,
                                            'Number of source models': num_source_models,
                                            'Fraction of sampled points': fraction_sampled_points,
                                            'Average correctness seen (mean)': df['Seen correct'].mean(),
                                            'Average correctness unseen (mean)': df['Unseen correct'].mean()})
                    for split in ['Seen', 'Unseen']:
                        mde, mde_lower, mde_upper = get_mde_with_error_bars(df, split=split)
                        #print(benchmark, num_source_models, fraction_sampled_points, technique, mde_lower, mde, mde_upper)
                        tidy_data_mde.append({'Technique': technique,
                                                'Number of source models': num_source_models,
                                                'Split': split.lower(),
                                                'Fraction of sampled points': fraction_sampled_points,
                                                'MDE': mde,
                                                'MDE lower CI': mde_lower,
                                                'MDE upper CI': mde_upper,})
        df_correctness = pd.DataFrame(tidy_data_correctness)
        df_mde = pd.DataFrame(tidy_data_mde)
        full_estimation_df = pd.DataFrame(all_tidy_results_estimation)
        full_estimation_df['Technique'] = full_estimation_df['Technique'].map(lambda t: technique_to_display_name[t])
        df_mde['Technique'] = df_mde['Technique'].map(lambda t: technique_to_display_name[t])
        df_correctness['Technique'] = df_correctness['Technique'].map(lambda t: technique_to_display_name[t])
        df_correctness.to_csv(cached_correctness_results_fn)
        df_mde.to_csv(cached_mde_results_fn)
        full_estimation_df.to_csv(cached_estimation_results_fn)