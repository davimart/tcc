""" Time comparison of different pre-processing reweighting techniques used in fsv_exp_fair_benchmarks.py

The output is a Tensor with 3 dimensions: number of pre-processing techniques, size of dataset, number of repetitions.
On each ijk cell will be the time to run a given preprocessing technique on a given dataset size in the k-th repetition.
"""
from numba import njit, prange
import numpy as np
import time
import os
import argparse
import pickle as pkl

from tqdm import tqdm
from utils.IFLiLiu.weights_fns import get_IF_weights
from utils.labelbias import jiang_weights
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from aif360.datasets import  StandardDataset

from aif360.algorithms.preprocessing import OptimPreproc, Reweighing
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import get_distortion_adult
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult

from fairSV.fair_shapley_sklearn import get_SV_matrix_numba_memory

# Settings the warnings to be ignored
import warnings

warnings.filterwarnings('ignore')



def change_aif360_dataset_size(aif360dataset, new_size):

    n = aif360dataset.features.shape[0]

    # Select new_size samples if new_size < n or duplicate samples if new_size > n
    if new_size<n:
        idx = np.array(np.random.choice(n, new_size, replace=False)).astype(int)
    else:
        idx = np.array(np.random.choice(n, new_size, replace=True)).astype(int)
    

    new_aif360dataset = aif360dataset.copy()
    new_aif360dataset.features = aif360dataset.features[idx]
    new_aif360dataset.labels = aif360dataset.labels[idx]
    new_aif360dataset.protected_attributes = aif360dataset.protected_attributes[idx]
    new_aif360dataset.instance_weights = aif360dataset.instance_weights[idx]
    new_aif360dataset.instance_names = list(np.array(aif360dataset.instance_names)[idx])
    new_aif360dataset.scores = aif360dataset.scores[idx]

    return new_aif360dataset

def change_aif360_feature_size(aif360dataset, new_size):
    """Change number of features of aif360 dataset.

    Args:
        aif360dataset (_type_): aif360 dataset
        new_size (_type_): new feature size
    """
    new_aif360dataset = aif360dataset.copy()
    if new_size < aif360dataset.features.shape[1]:
        new_aif360dataset.features = aif360dataset.features[:, :new_size]
        new_aif360dataset.feature_names = aif360dataset.feature_names[:new_size]
    else:
        new_aif360dataset.features = np.hstack((aif360dataset.features, np.random.rand(aif360dataset.features.shape[0], new_size-aif360dataset.features.shape[1])))
    return new_aif360dataset

#############################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument(
        "--features",
        default=18,
        type=int,
        help="number of features",
        required=True
)

exp_time = time.strftime("%Y%m%d-%H%M%S")
#############################################################################################################

techniques = ['LiLiu'] #['FairShap', 'simple', 'LabelBias', 'OptPrep']
dataset_sizes = [1000, 2000, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 80000, 100000]
N_repetitions = 10
N_feat = int(parser.parse_args().features)

#Create result folder based on the number of features
result_folder = f"results{os.sep}time{os.sep}F{N_feat}_{exp_time}"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)



results_time = np.zeros((len(techniques), len(dataset_sizes), N_repetitions))

SV = get_SV_matrix_numba_memory(np.random.rand(20,5), np.random.rand(10,5), np.array([1]*20), np.array([1]*10), 5)

pbar_dataset = tqdm(dataset_sizes)
for dataset in pbar_dataset:
    pbar_dataset.set_description(f"Dataset size: {dataset}")
    pbar_tech = tqdm(techniques, leave=False)
    for technique in pbar_tech:
        pbar_tech.set_description(f"Technique: {technique}")
        for iteration in tqdm(range(N_repetitions), leave=False):
            if technique == 'OptPrep':
                start_aif = time.time()
                privileged_groups = [{'race': 1}]
                unprivileged_groups = [{'race': 0}]
                dataset_orig_train = load_preproc_data_adult(['race'])
                optim_options = {
                    "distortion_fun": get_distortion_adult,
                    "epsilon": 0.05,
                    "clist": [0.99, 1.99, 2.99],
                    "dlist": [.1, 0.05, 0]
                }
                
                dataset_orig_train = change_aif360_dataset_size(dataset_orig_train, dataset)  
                #dataset_orig_train = change_aif360_feature_size(dataset_orig_train, N_feat)    
                end_aif = time.time()
                time_aif = end_aif - start_aif
                #assert dataset_orig_train.features.shape == (dataset, N_feat)
        
            else:
                N_tr = int(0.9 * dataset)
                N_tst = int(0.1 * dataset)

                x_tr = np.random.rand(N_tr,N_feat)
                x_tst = np.random.rand(N_tst,N_feat)

                y_tr = np.random.choice([0,1], size=N_tr, p=[.65, .35])
                y_tst = np.random.choice([0,1], size=N_tst, p=[.65, .35])

                a_tr = np.random.choice([0,1], size=N_tr, p=[.65, .35])
                a_tst = np.random.choice([0,1], size=N_tst, p=[.65, .35])

                privileged_groups = [{'a': 1}]
                unprivileged_groups = [{'a': 0}]


                # transform data to aif360 format
                start_aif = time.time()
                # Create a StandardDataset from the generated data
                dataset_orig_train = StandardDataset(
                    df=pd.DataFrame(np.hstack((x_tr, y_tr.reshape(-1,1), a_tr.reshape(-1,1))), columns=[f'x{i}' for i in range(N_feat)] + ['y', 'a']),
                    label_name='y', protected_attribute_names=['a'], favorable_classes=[1], privileged_classes=[[1]])
                dataset_orig_test = StandardDataset(
                    df=pd.DataFrame(np.hstack((x_tst, y_tst.reshape(-1,1), a_tst.reshape(-1,1))), columns=[f'x{i}' for i in range(N_feat)] + ['y', 'a']),
                    label_name='y', protected_attribute_names=['a'], favorable_classes=[1], privileged_classes=[[1]])
                end_aif = time.time()
                time_aif = end_aif - start_aif            

            if technique == 'FairShap':
                start = time.time()
                SV = get_SV_matrix_numba_memory(x_tr, x_tst, y_tr, y_tst, 5)
                end = time.time()
                results_time[techniques.index(technique), dataset_sizes.index(dataset), iteration] = end - start
                del SV

            elif technique == 'LabelBias':
                start = time.time()
                w_lbl_bias = jiang_weights(x_tr, y_tr, [a_tr], 1, n_iters=200)
                end = time.time()
                results_time[techniques.index(technique), dataset_sizes.index(dataset), iteration] = end - start

            
            elif technique == "simple":
                start = time.time()
                RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
                RW.fit(dataset_orig_train)
                dataset_orig_train = RW.transform(dataset_orig_train)
                end = time.time()
                results_time[techniques.index(technique), dataset_sizes.index(dataset), iteration] = end - start + time_aif
                
            elif technique == "OptPrep":
                start = time.time()
                OP = OptimPreproc(OptTools, optim_options,
                    unprivileged_groups = unprivileged_groups,
                    privileged_groups = privileged_groups)
                OP = OP.fit(dataset_orig_train)
                # Transform training data and align features
                dataset_transf_train = OP.transform(dataset_orig_train, transform_Y=True)
                end = time.time()
                results_time[techniques.index(technique), dataset_sizes.index(dataset), iteration] = end - start + time_aif
            elif  technique =="LiLiu":
                start = time.time()
                w = get_IF_weights(x_tr, y_tr, x_tst, y_tst, a_tst, 'eop',
                                   l2_reg=10, seed=41, alpha=1, beta=0, gamma=0)
                end = time.time()
                results_time[techniques.index(technique), dataset_sizes.index(dataset), iteration] = end - start + time_aif

        #save results_time to pickle
        with open(result_folder+'/results_time.pickle', 'wb') as f:
            pkl.dump(results_time, f)

#get results_time of 2 dim doing the mean of the n_iterations: dimensions are techniques and dataset_size
mean_results = np.mean(results_time, axis=2) #mean of the n_iterations
std_results = np.std(results_time, axis=2) #std of the n_iterations
assert mean_results.shape == (len(techniques), len(dataset_sizes))


#mean and std of result_time to pandas dataframe
df_mean = pd.DataFrame(mean_results, index=techniques, columns=dataset_sizes)
df_std = pd.DataFrame(std_results, index=techniques, columns=dataset_sizes)

#save to csv
df_mean.to_csv(result_folder+'/mean_time.csv')
df_std.to_csv(result_folder+'/std_time.csv')

# Create a pandas DataFrame to store the results
results_df = pd.DataFrame({
    'Technique': np.repeat(techniques, len(dataset_sizes) * N_repetitions),
    'Dataset_Size': np.tile(np.repeat(dataset_sizes, N_repetitions), len(techniques)),
    'Computation_Time': results_time.flatten()
})

# Plot the line plot using Seaborn
plt.figure(figsize=(10, 5))
sns.set_context("paper", font_scale=1.5)
plt.xlabel('Dataset size')
plt.ylabel('Time (s)')
plt.grid(True, linestyle='--', alpha=0.7)

sns.lineplot(x='Dataset_Size', y='Computation_Time', hue='Technique',
             data=results_df, ci='sd', linewidth=3)
plt.savefig(result_folder+'/time_comparison.png')

# Plot the line plot using Seaborn
plt.figure(figsize=(10, 5))
sns.set_context("paper", font_scale=1.5)
plt.xlabel('Dataset size')
plt.ylabel('Time (s)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.yscale('symlog')

sns.lineplot(x='Dataset_Size', y='Computation_Time', hue='Technique',
             data=results_df, ci='sd', linewidth=3)
plt.savefig(result_folder+'/time_comparison_log.png')

#Plot only for FairShap
plt.figure(figsize=(10, 5))
sns.set_context("paper", font_scale=1.5)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Dataset size')
plt.ylabel('Time (s)')
sns.lineplot(x='Dataset_Size', y='Computation_Time', hue='Technique',
             data=results_df[results_df['Technique'] == 'FairShap'],
             ci='sd',
             linewidth=3)
plt.savefig(result_folder+'/time_fairshap.png')


