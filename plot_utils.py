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

mde_correctness = 0.8

# TODO rewrite and give gemini credit
def bootstrap_mean_ci(data, B=1000, confidence_level=0.95):
    """
    Calculates the bootstrap mean and 95% confidence interval.

    Args:
    data: A list or numpy array of the sample data.
    B: The number of bootstrap samples to generate.
    confidence_level: The desired confidence level (default is 0.95).

    Returns:
    A tuple containing:
        - The bootstrap mean.
        - The lower and upper bounds of the confidence interval.
    """

    # 1. Calculate the sample mean and standard deviation
    sample_mean = np.mean(data)
    # sample_std = np.std(data, ddof=1)

    # 2. Generate bootstrap samples and calculate the mean for each
    bootstrap_means = []
    for _ in range(B):
        # Generate a bootstrap sample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # Calculate the mean of the bootstrap sample
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)

    # 3. Determine the confidence interval
    lower_bound = np.quantile(bootstrap_means, (1 - confidence_level) / 2)
    upper_bound = np.quantile(bootstrap_means, 1 - (1 - confidence_level) / 2)

    return sample_mean, lower_bound, upper_bound

def get_mde_with_error_bars(df, resolution=0.5, split='Seen', mdad_threshold=0.8):
    round_multiplier = 1/resolution * 100
    df.dropna()
    # first bin the data
    data = zip(df[f'{split} full accuracy difference'].tolist(), df[f'{split} correct'].tolist())
    bins = sorted(list(set(df[f'{split} full accuracy difference'].tolist())))
    bins = [math.floor(abs(x) * round_multiplier) / round_multiplier for x in bins]
    bin_to_data = defaultdict(list)
    for x, y in data:
        bin_to_data[x].append(y)
    mean_correctness_in_order = []
    lows_in_order = []
    ups_in_order = []
    first_m_over_80 = 0.5
    first_l_over_80 = 0.5
    first_u_over_80 = 0.5
    for x in bins:
        if len(bin_to_data) == 0:
            mean_correctness_in_order.append(0)
        m, l, u = bootstrap_mean_ci(bin_to_data[x])
        mean_correctness_in_order.append(m)
        lows_in_order.append(l)
        ups_in_order.append(u)
        if m >= mdad_threshold and first_m_over_80 == 0.5:
            first_m_over_80 = x
        if l >= mdad_threshold and first_l_over_80 == 0.5:
            first_l_over_80 = x
        if u >= mdad_threshold and first_u_over_80 == 0.5:
            first_u_over_80 = x

    # top error bar becomes bottom error bar for mde
    return first_m_over_80, first_u_over_80, first_l_over_80

palette = {
    'Random (uniform)': '#80b1d3',
    'Random (subtask stratified, equal)': '#bebada',
    'Random (subtask stratified, proportional)': '#fccde5',
    'Anchor Points': '#fdb462',
    'Anchor Points (predictor)': '#b3de69',
    'Diversity': '#8dd3c7',
    'tinyBenchmarks': '#fb8072',
    'Stratified sampling (confidence)': '#bc80bd',
}
technique_to_display_name = {
    "Random": 'Random (uniform)',
    "Random_Subtask_Stratified_Equal": 'Random (subtask stratified, equal)',
    "Random_Subtask_Stratified_Proportional": 'Random (subtask stratified, proportional)',
    "Anchor_Points_Weighted": 'Anchor Points',
    "Anchor_Points_Predictor": 'Anchor Points (predictor)',
    "DPP": 'Diversity',
    "tinyBenchmarks": 'tinyBenchmarks',
    'Stratified_Random_Sampling': 'Stratified sampling (confidence)',
}

benchmark_to_ylim_estimation = {
    'gpqa': (0, 0.18),
    'bbh': (0, 0.18),
    'mmlu-pro': (0, 0.18),
    'mmlu': (0, 0.18),
}
benchmark_to_ylim_kendall_tau = {
    'gpqa': (0, 1),
    'bbh': (0, 1),
    'mmlu-pro': (0, 1),
    'mmlu': (0, 1),
}
benchmark_to_ylim_mde = {
    'gpqa': (0, 0.3),
    'bbh': (0, 0.3),
    'mmlu-pro': (0, 0.3),
    'mmlu': (0, 0.3),
}
benchmark_to_display_name = {
    'gpqa': 'GPQA',
    'bbh': 'BBH',
    'mmlu-pro': 'MMLU-Pro',
    'mmlu': 'MMLU',
}
benchmark_to_full_seen_size = {
    'gpqa': 224,
    'bbh': 2880,
    'mmlu-pro': 6012,
    'mmlu': 5306,
}

number_to_highlight_color = {
    10: '#f4cccc',
    25: '#fce5cd',
    50: '#fff2cc',
    100: '#d9ead3',
    200: '#d0e0e3',
    250: '#d0e0e3',
    500: '#c9daf8',
    1000: '#d9d2e9'
}

def make_tidy_results(technique, ds, num_medoids, num_medoids_fraction, num_source_models, run_idx,
                      target_models_accuracies_seen, target_models_accuracies_unseen,
                      true_target_model_scores, estimated_scores, resolution=0.5):
    round_multiplier = 1/resolution * 100
    tidy_results = []
    model_comparison_pairs = list(itertools.combinations(range(len(target_models_accuracies_seen)), 2))
    num_comparisons = len(model_comparison_pairs)
    for i, (target_model_idx_A, target_model_idx_B) in enumerate(model_comparison_pairs):
        seen_acc_A = target_models_accuracies_seen[target_model_idx_A]
        seen_acc_B = target_models_accuracies_seen[target_model_idx_B]
        seen_acc_diff = seen_acc_A - seen_acc_B
        acc_A = true_target_model_scores[target_model_idx_A]
        acc_B = true_target_model_scores[target_model_idx_B]
        true_acc_diff = acc_A - acc_B
        unseen_acc_A = target_models_accuracies_unseen[target_model_idx_A]
        unseen_acc_B = target_models_accuracies_unseen[target_model_idx_B]
        unseen_acc_diff = unseen_acc_A - unseen_acc_B
        estimated_acc_diff = estimated_scores[target_model_idx_A] - estimated_scores[target_model_idx_B]
        
        seen_correct = int(math.copysign(1, estimated_acc_diff) == math.copysign(1, seen_acc_diff) and estimated_acc_diff != 0)
        seen_rounded_acc_diff = math.floor(abs(seen_acc_diff) * round_multiplier) / round_multiplier
        unseen_correct = int(math.copysign(1, estimated_acc_diff) == math.copysign(1, unseen_acc_diff) and estimated_acc_diff != 0)
        unseen_rounded_acc_diff = math.floor(abs(unseen_acc_diff) * round_multiplier) / round_multiplier

        result = {'Dataset': ds,
                  'Number of sampled points': num_medoids,
                  'Fraction of sampled points': num_medoids_fraction,
                  'Technique': technique,
                  'Number of source models': num_source_models,
                  'True acc diff': true_acc_diff, # on the seen + unseen set
                  'Seen acc diff': seen_acc_diff, # on the full seen set the microbenchmark was constructed from
                  'Unseen acc diff': unseen_acc_diff, # on the UNSEEN set
                  'Estimated acc diff': estimated_acc_diff, # just on the microbenchmark 
                  'Seen full accuracy difference': seen_rounded_acc_diff,
                  'Unseen full accuracy difference': unseen_rounded_acc_diff,
                  'Seen correct': seen_correct,
                  'Unseen correct': unseen_correct}
        
        tidy_results.append(result)
    return tidy_results

def make_tidy_results_estimation(technique, ds, num_medoids, num_medoids_fraction, num_source_models, run_idx,
                      target_models_accuracies_seen,
                      target_models_accuracies_unseen, 
                      target_models_accuracies_combined,
                      estimated_scores,
                      ):

    mean_seen_error = np.mean(np.abs(np.array(target_models_accuracies_seen) - np.array(estimated_scores)))
    mean_unseen_error = np.mean(np.abs(np.array(target_models_accuracies_unseen) - np.array(estimated_scores)))
    corr_seen = kendalltau(target_models_accuracies_seen, estimated_scores).correlation
    corr_unseen = kendalltau(target_models_accuracies_unseen, estimated_scores).correlation
    
    return {'Dataset': ds,
            'Number of sampled points': num_medoids,
            'Fraction of sampled points': num_medoids_fraction,
            'Number of source models': num_source_models,
            'Technique': technique,
            'Mean estimation error against seen accuracies': mean_seen_error,
            'Mean estimation error against unseen accuracies': mean_unseen_error,
            'Kendall tau correlation against seen accuracies': corr_seen,
            'Kendall tau correlation against unseen accuracies': corr_unseen,
            }

split_to_linestyle = {'seen': '-',
                      'unseen': '--'}
split_to_marker = {'seen': 'o',
                   'unseen': 's'}

