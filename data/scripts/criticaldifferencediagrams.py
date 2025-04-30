import os
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
from matplotlib.cm import viridis
from matplotlib.colors import Normalize
import lightgbm as lgb
import scikit_posthocs as sp

viewsingle = False

pd.set_option('display.max_columns', None)
def create_critical_difference_diagram(results, result2, name):
    if viewsingle:
        results_df = pd.DataFrame(results)
    else:
        results_df = pd.DataFrame(result2)
    # # https://github.com/hfawaz/cd-diagram/tree/master could also be an approach however it looks uglier and has different test.
    # # cd.draw_cd_diagram(results_df)
    # # TODO replace with https://github.com/mirkobunse/critdd this one can also have more lines!
    matrix = results_df.pivot(index='dataset_name', columns='classifier_name', values='accuracy')
    test_results = sp.posthoc_conover_friedman(matrix)
    plt.figure(figsize=(10, 2), dpi=100)
    # plt.title('Critical difference diagram of average score ranks')
    avg_rank = results_df.groupby('dataset_name').accuracy.rank(pct=True).groupby(results_df.classifier_name).mean()
    sp.critical_difference_diagram(avg_rank, test_results)
    plt.savefig('figures/figur' + str(name) + '_viewsingle_' + str(viewsingle) +'.png')
    plt.show()

results_list_memory = []
results_list_accuracy = []
results_list_memory_2 = []
results_list_accuracy_2 = []
results_list_tradeoff = []
results_list_tradeoff_2 = []
datasets = ['concrete', 'cycle', 'magic', 'spambase', 'statlog', 'superconductor']
model_counters = {'Pruned': 0, 'CEGB': 0, 'Pre-Pruned':0,
              'Quant-16': 0,
              'Quant-8': 0,
              'Merging': 0}
for data in datasets:
    df = pd.read_csv('collectivedata/' + f'{data}' + '/collectivedatauno.csv', sep=';', index_col=False)
    print(df)
    df['Trade-Off'] = df['accuracy'] / df['returnMemory']
    pruned = df[(df['enable_pruning'] == True) & (df['enable_prepruning'] == False) & (df['enable_cegb'] == False) & (df['enable_merging'] == False)& (df['enable_quantization'] == False)]
    pre_pruned = df[(df['enable_prepruning'] == True) & (df['enable_pruning'] == False) & (df['enable_cegb'] == False)& (df['enable_merging'] == False)& (df['enable_quantization'] == False)]
    cegb = df[(df['enable_cegb'] == True) & (df['enable_pruning'] == False) & (df['enable_prepruning'] == False) & (df['enable_merging'] == False)& (df['enable_quantization'] == False)]
    quant8 = df[(df['quantization_bits'] == 8) & (df['enable_quantization'] == True)]
    quant16 = df[(df['quantization_bits'] == 16) & (df['enable_quantization'] == True)]
    merging = df[(df['enable_merging'] == True) & (df['enable_pruning'] == False) & (df['enable_prepruning'] == False) & (df['enable_cegb'] == False) & (df['enable_quantization'] == False)]

    models = {'Pruned': pruned,
              'CEGB': cegb,
              'Pre-Pruned': pre_pruned,
              'Quant-16': quant16,
              'Quant-8': quant8,
              'Merging': merging}
    pruned = df[(df['enable_pruning'] == True)]
    pre_pruned = df[(df['enable_prepruning'] == True) ]
    cegb = df[(df['enable_cegb'] == True)]
    quant8 = df[(df['quantization_bits'] == 8) & (df['enable_quantization'] == True)]
    quant16 = df[(df['quantization_bits'] == 16) & (df['enable_quantization'] == True)]
    merging = df[(df['enable_merging'] == True)]
    models2 = {'Pruned': pruned,
              'CEGB': cegb,
              'Pre-Pruned': pre_pruned,
              'Quant-16': quant16,
              'Quant-8': quant8,
              'Merging': merging}
    for key in models:
        df = models[key]
        best_row = df.loc[df['accuracy'].idxmax()]  # Select the entire row
        best_row_mem = best_row['returnMemory'] # Select the entire row
        meanvalue_acc = 0
        meanvalue_mem = 0
        meanvalue_to = 0
        counter = 0
        for index, row in df.iterrows():
            meanvalue_acc += row['accuracy']
            meanvalue_mem += row['returnMemory']
            meanvalue_to += row['Trade-Off']
            counter += 1
        model_counters[key] += counter
        results_list_memory.append({'classifier_name': key, 'dataset_name': data, 'accuracy': meanvalue_mem/counter})
        results_list_accuracy.append({'classifier_name': key, 'dataset_name': data, 'accuracy': meanvalue_acc/counter})
        results_list_tradeoff.append({'classifier_name': key, 'dataset_name': data, 'accuracy': meanvalue_to/counter})

    for key in models2:
        df = models2[key]
        best_row = df.loc[df['accuracy'].idxmax()]  # Select the entire row
        best_row_mem = best_row['returnMemory'] # Select the entire row
        meanvalue_acc = 0
        meanvalue_mem = 0
        meanvalue_to = 0
        counter = 0
        for index, row in df.iterrows():
            meanvalue_acc += row['accuracy']
            meanvalue_mem += row['returnMemory']
            meanvalue_to += row['Trade-Off']
            counter += 1
        model_counters[key] += counter
        results_list_memory_2.append({'classifier_name': key, 'dataset_name': data, 'accuracy': meanvalue_mem/counter})
        results_list_accuracy_2.append({'classifier_name': key, 'dataset_name': data, 'accuracy': meanvalue_acc/counter})
        results_list_tradeoff_2.append({'classifier_name': key, 'dataset_name': data, 'accuracy': meanvalue_to/counter})

print(model_counters)
create_critical_difference_diagram(results_list_memory, results_list_memory_2, 'memory')
create_critical_difference_diagram(results_list_accuracy, results_list_accuracy_2, 'accuracy')
create_critical_difference_diagram(results_list_tradeoff, results_list_tradeoff_2, 'Trade-Off')
