import os
import argparse
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

def reformat_df(df, cols):
    df2 = pd.DataFrame(columns=cols)

    arduino_cols = ['arduino:avr:leonardo', 'arduino:samd:mkr1000', 'arduino:samd:mkrgsm1400', 'esp32:esp32:adafruit_feather_esp32_v2',
            'esp32:esp32:sparklemotion']
    columns = arduino_cols + ['arduino:avr:uno']
    othercols = [col for col in cols if col not in columns]
    for col in othercols:
        df2[col] = df[col]
    df2['arduino:avr:uno'] = df['returnMemory']

    for col in arduino_cols:
        df2[col] = 0
    return df2

def main():
    parser = argparse.ArgumentParser(description="Process some input arguments.")
    parser.add_argument('--df', type=str, required=True,
                        help='destination')
    parser.set_defaults(train=True)

    # Parse the command line arguments
    args = parser.parse_args()

    destination = args.df

    datasets = ['magic', 'spambase', 'statlog']
    cols = ['cegb_penalty_split',
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
            'arduino:avr:uno',
            'arduino:avr:leonardo',
            'arduino:samd:mkr1000',
            'arduino:samd:mkrgsm1400',
            'esp32:esp32:adafruit_feather_esp32_v2',
            'esp32:esp32:sparklemotion',
            'Score']
    dir = 'collectivedata'
    for data in datasets:
        if os.path.isfile(dir + '/' + data + '/collectivedatauno.csv') == True:
            grounddf = pd.read_csv(dir + '/' + data + '/collectivedatauno.csv', sep=';', index_col=False)
            if 'returnMemory' in grounddf.columns:
                grounddf = reformat_df(grounddf, cols)
        else:
            grounddf = pd.DataFrame(columns=cols)

        file_path = f'{destination}/{data}/*_i100*/print.csv'
        files = glob.glob(file_path, recursive=True)
        grounddf = grounddf[(grounddf['arduino:avr:uno'] != 999999) & (~grounddf['arduino:avr:uno'].isna())]

        for file in files:
            df = pd.read_csv(file, sep=';', index_col=False)
            grounddf = pd.concat([grounddf, df], ignore_index=True)
            if 'returnMemory' in df.columns:
                grounddf = reformat_df(grounddf, cols)
            grounddf = grounddf[(grounddf['arduino:avr:uno'] != 999999) & (~grounddf['arduino:avr:uno'].isna())]

        min_memory = grounddf['arduino:avr:uno'].min()
        max_memory = grounddf['arduino:avr:uno'].max()
        min_accuracy = grounddf['accuracy'].min()
        max_accuracy = grounddf['accuracy'].max()
        for index, row in grounddf.iterrows():
            grounddf.at[index, 'Score'] = calculate_score(row['accuracy'], row['arduino:avr:uno'], min_accuracy, max_accuracy, min_memory, max_memory)

        if os.path.isdir(dir) == False:
            os.mkdir(dir)
        if os.path.isdir(dir + '/' + data) == False:
            os.mkdir(dir + '/' + data)
        grounddf = grounddf.drop_duplicates()
        grounddf.to_csv(dir + '/' + data + '/collectivedata.csv', sep=';', index=False)

if __name__ == "__main__":
    main()
