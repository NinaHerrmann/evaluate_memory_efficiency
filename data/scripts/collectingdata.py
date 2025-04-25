import os

import pandas as pd
import glob
pd.set_option('display.max_columns', None)

def calculate_score(accuracy, memory, min_accuracy, max_accuracy, min_memory, max_memory):
    """
    Berechnet den Score basierend auf der Genauigkeit und dem Speicherbedarf.

    Parameters:
    accuracy (float): Aktuelle Genauigkeit
    memory (float): Aktueller Speicherbedarf
    min_accuracy (float): Minimale Genauigkeit
    max_accuracy (float): Maximale Genauigkeit
    min_memory (float): Minimale Speichergröße
    max_memory (float): Maximale Speichergröße

    Returns:
    float: Der berechnete Score
    """
    accuracy_weight = 0.5  # Gewichtung für die Genauigkeit
    memory_weight = 0.5  # Gewichtung für den Speicher

    normalized_accuracy = (accuracy - min_accuracy) / (max_accuracy - min_accuracy)
    normalized_memory = (min_memory - memory) / (min_memory - max_memory)

    score = accuracy_weight * normalized_accuracy + memory_weight * normalized_memory
    return score

datasets = ['concrete', 'cycle', 'magic', 'spambase', 'statlog', 'superconductor']
model_counters = {'Pruned': 0, 'CEGB': 0, 'Pre-Pruned':0,
              'Quant-16': 0,
              'Quant-8': 0,
              'Merging': 0}
# create an empty df ith the column cegb_penalty_split;enable_cegb;enable_merging;enable_prepruning;enable_pruning;enable_quantization;lambda_l1;lambda_l2;learning_rate;max_depth;min_data_in_leaf;min_gain_to_split;n_estimators;num_clusters;num_leaves;pruning_percentile;quantization_bits;quantization_diff;quantization_type;stopping_rounds;rank;train_score;test_score;estimateMemory;accuracy;returnMemory
cols = [
    'cegb_penalty_split',
    'enable_cegb',
    'enable_merging',
    'enable_prepruning',
    'enable_pruning',
    'enable_quantization',
    'lambda_l1',
    'lambda_l2',
    'learning_rate',
    'max_depth',
    'min_data_in_leaf',
    'min_gain_to_split',
    'n_estimators',
    'num_clusters',
    'num_leaves',
    'pruning_percentile',
    'quantization_bits',
    'quantization_diff',
    'quantization_type',
    'stopping_rounds',
    'rank',
    'train_score',
    'test_score',
    'estimateMemory',
    'accuracy',
    'returnMemory'
]

# Erstellen eines leeren DataFrames mit den definierten Spalten
for data in datasets:
    grounddf = pd.DataFrame(columns=cols)

    file_path = f'../{data}/*_i1000*/print.csv'
    files = glob.glob(file_path, recursive=True)

    for file in files:
        # Lies die CSV-Datei
        df = pd.read_csv(file, sep=';', index_col=False)
        # merge the grounddf variable and the df variable
        grounddf = pd.concat([grounddf, df], ignore_index=True)
    # check if directory exists
    min_memory = grounddf['returnMemory'].min()
    max_memory = grounddf['returnMemory'].max()
    min_accuracy = grounddf['accuracy'].min()
    max_accuracy = grounddf['accuracy'].max()
    for index, row in grounddf.iterrows():
        grounddf.at[index, 'Score'] = calculate_score(row['accuracy'], row['returnMemory'], min_accuracy, max_accuracy, min_memory, max_memory)
    print(grounddf)
    dir = 'collectivedata'

    if os.path.isdir(dir) == False:
        os.mkdir(dir)
    if os.path.isdir(dir + '/' + data) == False:
        os.mkdir(dir + '/' + data)

    grounddf.to_csv(dir + '/' + data + '/collectivedata.csv', sep=';', index=False)
    # if not create it


