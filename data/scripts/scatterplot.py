import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
import numpy as np
import os
import scikit_posthocs as sp
import argparse

pd.set_option('display.max_columns', None)

def create_critical_difference_diagram(results, name, viewsingle):
    results_df = pd.DataFrame(results)

    # # https://github.com/hfawaz/cd-diagram/tree/master could also be an approach however it looks uglier and has different test.
    # cd.draw_cd_diagram(results_df)
    # # TODO replace with https://github.com/mirkobunse/critdd this one can also have more lines!
    matrix = results_df.pivot(index='dataset_name', columns='classifier_name', values='accuracy')
    test_results = sp.posthoc_conover_friedman(matrix)
    plt.figure(figsize=(10, 2), dpi=100)
    # plt.title('Critical difference diagram of average score ranks')
    avg_rank = results_df.groupby('dataset_name').accuracy.rank(pct=True).groupby(results_df.classifier_name).mean()
    sp.critical_difference_diagram(avg_rank, test_results)
    if os.path.isdir(f'{destination}/CDD/{name}') == False:
        os.makedirs(f'{destination}/CDD/{name}')

    plt.savefig(f'{destination}/CDD/{name}/' + 'VS_' + str(viewsingle) +'.png')
    #plt.show()

def create_boxplot(models, column, name, dataset, viewMixed=False):
    listofdata = []
    for key in models:
        df = models[key]
        listofdata.append(df[column])
    fig, axs = plt.subplots()
    axs.boxplot(listofdata, labels=models.keys())
    if viewMixed:
        name = f'{destination}/boxplots/{dataset}/Mixed/'
    else:
        name = f'{destination}/boxplots/{dataset}/'
    if os.path.isdir(name) == False:
        os.makedirs(name)
    plt.savefig(name + str(column) + '.png')

def create_scatter_score(models, dataset, viewMixed=False):
    for key in models:
        df = models[key]
        rightcolumn = 'percentile'
        if key == 'Pruned':
            rightcolumn = 'pruning_percentile'
        if key == 'CEGB':
            rightcolumn = 'cegb_penalty_split'
        if key == 'Pre-Pruned':
            continue
        if key == 'Quant-16':
            continue
        if key == 'Quant-8':
            continue
        if key == 'Merging':
            rightcolumn = 'num_clusters'
        fig, ax1 = plt.subplots()
        ax1.scatter(df[rightcolumn], df['Score'], c='b')
        if viewMixed:
            name = f'{destination}/Scatter/Score/{dataset}/Mixed/'
        else:
            name = f'{destination}/Scatter/Score/{dataset}/'
        if os.path.isdir(name) == False:
            os.makedirs(name)
        plt.savefig(name + str(key) + '.png')
        #fig.show()
def create_scatter_diagram(models, dataset, viewMixed=False):
    for key in models:
        df = models[key]
        rightcolumn = 'percentile'
        if key == 'Pruned':
            rightcolumn = 'pruning_percentile'
        if key == 'CEGB':
            rightcolumn = 'cegb_penalty_split'
        if key == 'Pre-Pruned':
            continue
        if key == 'Quant-16':
            continue
        if key == 'Quant-8':
            continue
        if key == 'Merging':
            rightcolumn = 'num_clusters'
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.scatter(df[rightcolumn], df['arduino:avr:uno'], c='b')
        ax2.scatter(df[rightcolumn], df['accuracy'], c='r')
        if viewMixed:
            name = f'{destination}/Scatter/MemAcc/{dataset}/Mixed/'
        else:
            name = f'{destination}/Scatter/MemAcc/{dataset}/'
        if os.path.isdir(name) == False:
            os.makedirs(name)
        plt.savefig(name + str(key) + '.png')

        #fig.show()

parser = argparse.ArgumentParser(description="Process some input arguments.")
# Add --dataset, --type, and --train argument
parser.add_argument('--sf', type=str, required=True,
                    help='Number of iterations')
parser.add_argument('--df', type=str, required=True,
                    help='Number of random states')
parser.set_defaults(train=True)

# Parse the command line arguments
args = parser.parse_args()

source = args.sf
destination = args.df

datasets = ['magic', 'spambase', 'statlog']
model_counters = {'Pruned': 0, 'CEGB': 0, 'Pre-Pruned':0,
              'Quant-16': 0,
              'Quant-8': 0,
              'Merging': 0}
model_counterssingle = {'Pruned': 0, 'CEGB': 0, 'Pre-Pruned':0,
              'Quant-16': 0,
              'Quant-8': 0,
              'Merging': 0}
results_list_memory = []
results_list_accuracy = []
results_list_memory_2 = []
results_list_accuracy_2 = []
results_list_tradeoff = []
results_list_tradeoff_2 = []
for data in datasets:

    df = pd.read_csv(f'{source}/{data}' + '/collectivedata.csv', sep=';', index_col=False)

    min_memory = df['arduino:avr:uno'].min()
    max_memory = df['arduino:avr:uno'].max()
    min_accuracy = df['accuracy'].min()
    max_accuracy = df['accuracy'].max()

    pruned = df[(df['enable_pruning'] == True) & (df['enable_prepruning'] == False) & (df['enable_cegb'] == False) & (df['enable_merging'] == False)& (df['enable_quantization'] == False)]
    pre_pruned = df[(df['enable_prepruning'] == True) & (df['enable_pruning'] == False) & (df['enable_cegb'] == False)& (df['enable_merging'] == False)& (df['enable_quantization'] == False)]
    cegb = df[(df['enable_cegb'] == True) & (df['enable_pruning'] == False) & (df['enable_prepruning'] == False) & (df['enable_merging'] == False)& (df['enable_quantization'] == False)]
    quant8 = df[(df['quantization_bits'] == 8) & (df['enable_quantization'] == True) & (df['enable_prepruning'] == False) & (df['enable_cegb'] == False) & (df['enable_merging'] == False)]
    quant16 = df[(df['quantization_bits'] == 16) & (df['enable_quantization'] == True)& (df['enable_prepruning'] == False) & (df['enable_cegb'] == False) & (df['enable_merging'] == False)]
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

    create_boxplot(models, 'arduino:avr:uno', 'memory', data)
    create_boxplot(models, 'accuracy', 'memory', data)
    create_boxplot(models, 'Score', 'score', data)
    create_scatter_score(models, data)
    create_scatter_diagram(models, data)
    for key in models:
        df = models[key]
        # check if the df is empty
        if df.empty:
            continue
        print(key)
        print(df)
        best_row = df.loc[df['accuracy'].idxmax()]  # Select the entire row
        best_row_mem = best_row['arduino:avr:uno'] # Select the entire row
        meanvalue_acc = 0
        meanvalue_mem = 0
        meanvalue_to = 0
        counter = 0
        for index, row in df.iterrows():
            meanvalue_acc += row['accuracy']
            meanvalue_mem += row['arduino:avr:uno']
            meanvalue_to += row['Score']
            counter += 1
        model_counterssingle[key] += counter
        results_list_memory.append({'classifier_name': key, 'dataset_name': data, 'accuracy': meanvalue_mem/counter})
        results_list_accuracy.append({'classifier_name': key, 'dataset_name': data, 'accuracy': meanvalue_acc/counter})
        results_list_tradeoff.append({'classifier_name': key, 'dataset_name': data, 'accuracy': meanvalue_to/counter})

    for key in models2:
        df = models2[key]
        best_row = df.loc[df['accuracy'].idxmax()]  # Select the entire row
        best_row_mem = best_row['arduino:avr:uno'] # Select the entire row
        meanvalue_acc = 0
        meanvalue_mem = 0
        meanvalue_to = 0
        counter = 0
        for index, row in df.iterrows():
            meanvalue_acc += row['accuracy']
            meanvalue_mem += row['arduino:avr:uno']
            meanvalue_to += row['Score']
            counter += 1
        model_counters[key] += counter
        results_list_memory_2.append({'classifier_name': key, 'dataset_name': data, 'accuracy': meanvalue_mem/counter})
        results_list_accuracy_2.append({'classifier_name': key, 'dataset_name': data, 'accuracy': meanvalue_acc/counter})
        results_list_tradeoff_2.append({'classifier_name': key, 'dataset_name': data, 'accuracy': meanvalue_to/counter})
    create_boxplot(models2, 'arduino:avr:uno', 'memory', data, True)
    create_boxplot(models2, 'accuracy', 'memory', data, True)
    create_boxplot(models2, 'Score', 'score', data, True)
    create_scatter_score(models2, data, True)
    create_scatter_diagram(models2, data, True)
    #create_critical_difference_diagram(results_list_tradeoff, results_list_tradeoff_2, 'Score')
print(model_counterssingle)
print(model_counters)


create_critical_difference_diagram(results_list_memory, 'memory', True)
create_critical_difference_diagram(results_list_accuracy, 'accuracy', True)
create_critical_difference_diagram(results_list_tradeoff, 'Score', True)
create_critical_difference_diagram(results_list_memory_2, 'memory', False)
create_critical_difference_diagram(results_list_accuracy_2, 'accuracy', False)
create_critical_difference_diagram(results_list_tradeoff_2, 'Score', False)
