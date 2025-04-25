import argparse
import os
import pickle
from datetime import datetime
from itertools import combinations
from pathlib import Path
import random

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import linregress
from scipy.stats import randint, uniform
from sklearn.metrics import accuracy_score, root_mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from ucimlrepo import fetch_ucirepo

from RandomizedSearch import LightGBMRandomizedSearch, CustomScorer, SizeMetric, AccuracyImprovement, SizeImprovement, \
    ThresholdMetric, TreeMetric

def round_numbers(df):
    newdf = df
    numerical_cols = newdf.select_dtypes(include=['int64', 'float64']).columns
    newdf[numerical_cols] = newdf[numerical_cols].apply(lambda x: x.round(2))
    return newdf

def train_base_model(iterations, objective, fixed_hyperparameters, custom_hyperparameters, X_train, y_train, rand_state, num_class=6, threads=4,
                     estimate=False):
    """
    Trains the base LightGBM model using RandomizedSearchCV with custom scoring metrics for model performance
    and memory efficiency.

    Parameters:
        objective (str): The learning objective, must be one of "regression", "multiclass", or "binary"
        fixed_hyperparameters (dict): Dictionary of hyperparameter distributions for random search
        custom_hyperparameters (dict): Optimization technique hyperparameters to be combined with the best parameters found
        X_train (np.ndarray): array-like of shape (n_samples, n_features), training data.
        y_train (np.ndarray): array-like of shape (n_samples,), target values.
        num_class (int, optional): Number of classes for multiclass classification (Defaults to 6)
        threads (int, optional): Number of threads/cores used to train the LightGBM model
        estimate (bool, optional): Toggle whether to estimate model size during training (Defaults to False)

    Returns:
        model (LightGBMRandomizedSearch): The best trained LightGBM model
        base_score (float): Mean test accuracy/RMSE of the best model
        base_size (float): Mean memory usage of the best model
        combined (dict): Combined dictionary of best and custom hyperparameters
        search_random: The fitted RandomizedSearchCV object
    """

    if objective == "regression":
        scoring = {
            'memory_efficiency': CustomScorer(),
            'accuracy': 'r2_score',
            'memory': SizeMetric(),
        }
        search_random = RandomizedSearchCV(
            estimator=LightGBMRandomizedSearch(objective='regression', metric='r2_score', verbosity=-1, num_threads=threads,
                                               estimate_size=estimate),
            param_distributions=fixed_hyperparameters,
            n_iter=iterations,
            cv=2,
            n_jobs=1,
            random_state=rand_state,
            return_train_score=True,
            scoring=scoring,
            refit='memory_efficiency',
            verbose=4
        )
    elif objective == "multiclass":
        scoring = {
            'memory_efficiency': CustomScorer(),
            'accuracy': 'accuracy',
            'memory': SizeMetric(),
        }
        search_random = RandomizedSearchCV(
            estimator=LightGBMRandomizedSearch(objective='multiclass', num_class=num_class, metric='multi_logloss',
                                               verbosity=-1, num_threads=threads, estimate_size=estimate),
            param_distributions=fixed_hyperparameters,
            n_iter=iterations,
            cv=2,
            n_jobs=-1,
            random_state=rand_state,
            return_train_score=True,
            scoring=scoring,
            refit='memory_efficiency',
            verbose=4
        )
    elif objective == "binary":
        scoring = {
            'memory_efficiency': CustomScorer(),
            'accuracy': 'accuracy',
            'memory': SizeMetric(),
        }
        search_random = RandomizedSearchCV(
            estimator=LightGBMRandomizedSearch(objective='binary', num_class=1, metric='binary_logloss', verbosity=-1,
                                               num_threads=threads, estimate_size=estimate),
            param_distributions=fixed_hyperparameters,
            n_iter=iterations,
            cv=2,
            n_jobs=-1,
            random_state=rand_state,
            return_train_score=True,
            scoring=scoring,
            refit='memory_efficiency',
            verbose=4
        )
    search_random.fit(X_train, y_train)
    model = search_random.best_estimator_
    best_params = search_random.best_params_
    combined = {**custom_hyperparameters}
    combined.update({k: [v] for k, v in best_params.items()})

    base_score = search_random.cv_results_["mean_test_accuracy"][search_random.best_index_]
    base_size = search_random.cv_results_["mean_test_memory"][search_random.best_index_]
    return model, base_score, base_size, combined, search_random


def save_variables(settings: str, variables: dict, base_path: str) -> str:
    """
    Saves multiple variables to pickle files in a timestamped directory.

    Parameters:
        variables (dict): Dictionary of variable names and their values to save
        base_path (str): Base directory path where the timestamped folder will be created

    Returns:
        str: Path to the created folder containing the saved variables
    """
    folder = Path(base_path) / f"vars_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{settings}"
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in variables.items():
        with open(folder / f"{name}.pkl", "wb") as f:
            pickle.dump(value, f)
    return str(folder)


def load_variables(folder_path: str) -> dict:
    """
    Loads all pickle files from a specified folder into a dictionary.

    Parameters:
        folder_path (str): Path to the folder containing pickle files

    Returns:
        dict: Dictionary where keys are the filenames (without .pkl extension) and
              values are the loaded objects
    """
    return {f.stem: pickle.load(open(f, "rb"))
            for f in Path(folder_path).glob("*.pkl")}


def evaluate(results, X_train, y_train, X_test, y_test, path, random_state, top=10, task="binary", generate_outputs=True):
    """
    Evaluates the top performing model configurations based on memory efficiency.

    Parameters:
        results (dict): Dictionary containing RandomizedSearchCV results including metrics and parameters
        X_train (np.ndarray): array-like of shape (n_samples, n_features), training data.
        y_train (np.ndarray): array-like of shape (n_samples,), train target values.
        X_test (np.ndarray): array-like of shape (n_samples, n_features), testing data.
        y_test (np.ndarray): array-like of shape (n_samples,), test target values.
        top (int, optional): Number of top configurations to evaluate. Defaults to 10
        task (str, optional): The learning objective, must be one of "regression", "multiclass", or "binary"
        generate_outputs (bool, optional): Whether to generate model outputs to disk as C++ header files.

    Returns:
        List[LightGBMRandomizedSearch]: List of trained models for the top configurations
    """
    train_metric = results["mean_train_memory_efficiency"]
    test_accuracy = results["mean_test_accuracy"]
    params = results["params"]

    valid_indices = np.where(~np.isnan(train_metric))[0]
    valid_train_metric = train_metric[valid_indices]
    valid_test_accuracy = test_accuracy[valid_indices]
    valid_params = [params[i] for i in valid_indices]

    sorted_indices = np.argsort(valid_train_metric)[-top:][::-1]
    best_combinations = [(valid_params[i], valid_train_metric[i], valid_test_accuracy[i]) for i in sorted_indices]
    trained_models = []
    test_accuracy = []
    file_path = path + "/print.csv"
    first = True

    for rank, (config, train_score, test_score) in enumerate(best_combinations, 1):
        if task == "binary":
            model = LightGBMRandomizedSearch(objective='binary', num_class=1, metric='binary_logloss', verbosity=-1,
                                             **config)
        if task == "multiclass":
            model = LightGBMRandomizedSearch(objective='multiclass', num_class=7, metric='multi_logloss', verbosity=-1,
                                             **config)
        if task == "regression":
            model = LightGBMRandomizedSearch(objective='regression', metric='rmse', verbosity=-1, **config)

        model.fit(X_train, y_train)
        trained_models.append(model)
        file = Path(file_path)
        if first:
            with file.open('a') as fileopen:
                first = False
                platforms = ['arduino:avr:uno', 'arduino:avr:leonardo', 'arduino:samd:mkr1000',
                             'arduino:samd:mkrgsm1400', 'arduino:esp32:unowifi',
                             'esp32:esp32:adafruit_feather_esp32_v2', 'esp32:esp32:sparklemotion', 'arduino:samd:nano_33_ble']
                fileopen.write(";".join(config.keys()) + ";rank;train_score;test_score;estimateMemory;accuracy;random_state;")
                fileopen.write(";".join([str(v) for v in platforms]) + ";\n")

        if model is not None:
            if task == "binary" or task == "multiclass":
                accuracy = accuracy_score(y_test, model.predict(X_test))
            if task == "regression":
                accuracy = r2_score(y_test, model.predict(X_test)) # {accuracy:.4f};
            with file.open('a') as fileopen:
                fileopen.write(";".join([str(v) for v in config.values()]) + ";")
                fileopen.write(f"{rank};{train_score};{test_score:.4f};{model.model.estimateMemory()};{accuracy:.4f};{random_state};")
                fileopen.write(";".join([str(v) for v in model.model.returnMemory()]) + ";\n")

    if generate_outputs:
        for i, model in enumerate(trained_models):
            model.model.generate(path, str(i + 1))

    return trained_models


def analyze_grid_search(cv_results, path):
    """
    Analyzes grid search results of the quantization parameter configurations.
    Exports results to 'quantization_gridsearch.csv'

    Parameters:
        cv_results (dict): Dictionary of GridSearchCV results
    """
    results_df = pd.DataFrame(cv_results)
    metrics = ['mean_train_memory_efficiency', 'mean_test_accuracy', 'mean_test_memory']
    improvement = ['mean_test_accuracy_improvement', 'mean_test_memory_improvement']
    results_df = results_df.dropna(subset=metrics)
    hyperparameters = ['param_enable_quantization', 'param_enable_pruning', 'param_enable_merging',
                       'param_enable_cegb', 'param_enable_prepruning']
    param_quantization_bits = [8, 16]
    param_quantization_diff = ['leafs', 'thresholds', 'both']
    param_quantization_type = ['affine', 'scale']
    filtered_df = results_df.copy()
    analysis_results = []
    for bits in param_quantization_bits:
        for diff in param_quantization_diff:
            for q_type in param_quantization_type:
                combination = filtered_df[
                    (filtered_df['param_quantization_bits'] == bits) &
                    (filtered_df['param_quantization_diff'] == diff) &
                    (filtered_df['param_quantization_type'] == q_type)
                    ]
                if not combination.empty:
                    median_metrics = {
                        'mean_test_accuracy': combination['mean_test_accuracy'].iloc[0],
                        'mean_test_memory': combination['mean_test_memory'].iloc[0],
                        'mean_test_accuracy_improvement': combination['mean_test_accuracy_improvement'].iloc[0],
                        'mean_test_memory_improvement': combination['mean_test_memory_improvement'].iloc[0],
                        'mean_test_trees': combination['mean_test_trees'].iloc[0],
                        'mean_test_thresholds': combination['mean_test_thresholds'].iloc[0]
                    }

                    analysis_results.append({
                        'combination': (bits, diff, q_type),
                        **median_metrics
                    })
    analysis_results.append({
        'combination': "overall",
        'mean_test_accuracy': results_df['mean_test_accuracy'].median(),
        'mean_test_memory': results_df['mean_test_memory'].median(),
        'mean_test_accuracy_improvement': results_df['mean_test_accuracy_improvement'].median(),
        'mean_test_memory_improvement': results_df['mean_test_memory_improvement'].median(),
        'mean_test_trees': combination['mean_test_trees'].median(),
        'mean_test_thresholds': combination['mean_test_thresholds'].median()
    })
    analysis_results = pd.DataFrame(analysis_results)
    analysis_results.to_csv(path + '/quantization_gridsearch.csv', index=False)


def process_params(row):
    """
    Processes hyperparameter configuration for analysis. If a optimization method
    is not enabled, its corresponding hyperparameters are set to None.
    """
    if not row['param_enable_quantization']:
        row['param_quantization_type'] = None
        row['param_quantization_diff'] = None
        row['param_quantization_bits'] = 32
    else:
        type_map = {'affine': 1, 'scale': 2}
        row['param_quantization_type'] = type_map[row['param_quantization_type']]

        diff_map = {'leafs': 1, 'thresholds': 2, 'both': 3}
        row['param_quantization_diff'] = diff_map[row['param_quantization_diff']]
    if not row['param_enable_pruning']:
        row['param_pruning_percentile'] = None
    if row['param_num_clusters'] > np.ceil(row["mean_test_thresholds"]):
        row['param_num_clusters'] = np.ceil(row["mean_test_thresholds"])
    if not row['param_enable_merging']:
        row['param_num_clusters'] = None
    if not row['param_enable_cegb']:
        row['param_cegb_penalty_split'] = None
    if row['param_num_leaves'] >= 2 ** row["param_max_depth"]:
        row['param_num_leaves'] = 2 ** row["param_max_depth"] - 1
    if not row['param_enable_prepruning']:
        row['param_num_leaves'] = None
        row['param_max_depth'] = None
        row['param_min_data_in_leaf'] = None
        row['param_min_gain_to_split'] = None
    return row


def analyze_correlations(dataset, continuous_params, results_df, metrics, categorical_params, path):
    """
    Performs correlation analysis for continuous parameters and ANOVA tests for categorical parameters.
    Exports correlation analysis results to 'continuous.csv' and ANOVA test results to 'categorical.csv'.
    """
    correlation_df = pd.DataFrame(
        columns=['Parameter', 'Metric', 'SpearmanCorrelation', 'PearsonCorrelation', 'SpearmanCorrelation_p_value',
                 'PearsonCorrelation_p_value', 'Regression_Slope', 'Regression_Intercept', 'Regression_R2', ])
    anova_df = pd.DataFrame(columns=['Parameter', 'Metric', 'F_statistic', 'ANOVA_p_value'])

    for param in continuous_params:
        param_values = results_df[f'param_{param}']
        if param_values.empty:
            continue
        for metric in metrics:
            metric_values = results_df[metric]

            # Remove any error values
            valid_indices = metric_values <= 999998
            parameter_values = pd.to_numeric(param_values[valid_indices], errors='coerce').apply(
                lambda x: x if pd.notnull(x) else 0)
            metric_values = metric_values[valid_indices]
            spearmancorrelation, spearmancorr_p_value = stats.spearmanr(parameter_values, metric_values,
                                                                        nan_policy='omit')
            pearsoncorrelation, pearsoncorr_p_value = stats.pearsonr(parameter_values, metric_values)

            slope, intercept, r_value, p_value, std_err = linregress(parameter_values, metric_values)

            correlation_df = pd.concat([correlation_df, pd.DataFrame({
                'Parameter': [param],
                'Metric': [metric],
                'SpearmanCorrelation': [spearmancorrelation],
                'SpearmanCorrelation_p_value': [spearmancorr_p_value],
                'PearsonCorrelation': [pearsoncorrelation],
                'PearsonCorrelation_p_value': [pearsoncorr_p_value],
                'Regression_Slope': [slope],
                'Regression_Intercept': [intercept],
                'Regression_R2': [r_value],
                'Regression_p_value': [p_value],
            })], ignore_index=True)

    for param in categorical_params:
        param_values = results_df[f'param_{param}'].dropna()
        if param_values.empty:
            continue
        for metric in metrics:
            metric_values = results_df.loc[param_values.index, metric]

            groups = [metric_values[param_values == value] for value in param_values.unique()]
            try:
                f_statistic, p_value = stats.f_oneway(*groups)

                new_data = pd.DataFrame({
                    'Parameter': [param],
                    'Metric': [metric],
                    'F_statistic': [round(f_statistic, 3) if pd.notna(f_statistic) else np.nan],
                    'ANOVA_p_value': [f"{p_value}" if pd.notna(p_value) else np.nan]
                })

                if not new_data.isna().all(axis=None) and not new_data.empty:
                    anova_df = pd.concat([anova_df, new_data], ignore_index=True)
            except ValueError as e:
                print(f"Could not perform ANOVA for {param} on {metric}: {str(e)}")

    roundeddf = round_numbers(correlation_df)
    latex_table = roundeddf.to_latex(index=False)
    with open(f'{path}/{dataset}table_continous.tex', 'w') as f:
        f.write(latex_table)
    roundanova = round_numbers(anova_df)
    latex_table = roundanova.to_latex(index=False)
    with open(f'{path}/{dataset}_table_anovacat.tex', 'w') as f:
        f.write(latex_table)
    correlation_df.to_csv(path + '/continuous.csv', index=False)
    anova_df.to_csv(path + '/categorical.csv', index=False)


def analyze_improvement(results_df, path):
    """"
    Analyzes improvement metrics across different optimization techniques and their combinations.
    Saves results to 'improvement.csv'.
    """
    hyperparameters = ['param_enable_quantization', 'param_enable_pruning', 'param_enable_merging',
                       'param_enable_cegb', 'param_enable_prepruning',
                       ]

    analysis_results = []

    for r in range(1, len(hyperparameters) + 1):
        for combination in combinations(hyperparameters, r):
            filtered_df = results_df.copy()
            for param in hyperparameters:
                if param in combination:
                    filtered_df = filtered_df[filtered_df[param] == True]
                else:
                    filtered_df = filtered_df[filtered_df[param] == False]
            length = len(filtered_df)
            metrics_dict = {
                'combination': combination,
                'num_instances': len(filtered_df),
                'mean_test_accuracy': None,
                'mean_test_memory': None,
                'accuracy_improvement': None,
                'memory_improvement': None,
                'trees': None,
                'thresholds': None
            }

            if length > 0:
                metrics_dict.update({
                    'mean_test_accuracy': filtered_df['mean_test_accuracy'].median(),
                    'mean_test_memory': filtered_df['mean_test_memory'].median(),
                    'accuracy_improvement': filtered_df['mean_test_accuracy_improvement'].median(),
                    'memory_improvement': filtered_df['mean_test_memory_improvement'].median(),
                    'trees': filtered_df['mean_test_trees'].median(),
                    'thresholds': filtered_df['mean_test_thresholds'].median(),
                })

            analysis_results.append(metrics_dict)
    # print(analysis_results)
    filtered_results = [res for res in analysis_results if res is not None]
    analysis_results = pd.DataFrame(filtered_results)
    # analysis_results = pd.DataFrame(analysis_results)
    analysis_results.to_csv(path + '/improvement.csv', index=False)


def analyzeResults(dataset, cv_results, path):
    """
    Performs a statistical analysis of the random search results.

    Parameters:
        cv_results (dict): Dictionary of RandomizedSearchCV results
        path (str): Dictionary of parameter distributions used in search
    """
    results_df = pd.DataFrame(cv_results)
    metrics = ['mean_test_accuracy', 'mean_test_memory']
    improvement = ['mean_test_accuracy_improvement', 'mean_test_memory_improvement']
    results_df = results_df.dropna(subset=metrics)

    results_df = results_df.apply(process_params, axis=1)

    categorical_params = [
        'enable_quantization', 'quantization_type', 'quantization_diff',
        'quantization_bits', "enable_pruning", "enable_merging",
        'enable_cegb', 'enable_prepruning',
    ]
    continuous_params = [
        'min_gain_to_split', 'cegb_penalty_split', 'min_data_in_leaf',
        'pruning_percentile', "num_clusters", 'max_depth', "num_leaves"
    ]
    analyze_improvement(results_df, path)
    analyze_correlations(dataset, continuous_params, results_df, metrics, categorical_params, path)


threads = 4
estimate_size = False

fixed_hyperparameters = {
    "learning_rate": uniform(0.01, 0.8),
    "n_estimators": randint(20, 6000),
    "lambda_l1": uniform(0.01, 0.8),
    "lambda_l2": uniform(0.01, 0.8),
    "stopping_rounds": [10, 20, 50, 100],
}

custom_hyperparameters = {
    ### PREPRUNING HYPERPARAMETERS ###
    "enable_prepruning": [False, True],
    "max_depth": randint(2, 10),
    "num_leaves": randint(3, 1023),
    "min_gain_to_split": uniform(0.01, 0.2),
    "min_data_in_leaf": randint(21, 60),

    ### POSTPRUNING HYPERPARAMETERS ###
    "enable_pruning": [False, True],
    "pruning_percentile": uniform(0.01, 0.8),

    ### CEGB HYPERPARAMETERS ###
    "enable_cegb": [False, True],
    "cegb_penalty_split": uniform(0.01, 0.15),

    ### THRESHOLD SHARING HYPERPARAMETERS ###
    "enable_merging": [False, True],
    "num_clusters": randint(3, 25),

    ### QUANTIZATION HYPERPARAMETERS ###
    "enable_quantization": [False, True],
    "quantization_type": ["affine", "scale"],
    "quantization_diff": ["leafs", "thresholds", "both"],
    "quantization_bits": [8, 16],
}


def load_data(name, id, task="binary", num_class=1):
    data_folder = "data"
    X_file = os.path.join(data_folder, name + "X.npy")
    y_file = os.path.join(data_folder, name + "Y.npy")
    # Check if the dataset already exists
    if os.path.exists(X_file) and os.path.exists(y_file):
        X = np.load(X_file, allow_pickle=True)
        y = np.load(y_file, allow_pickle=True)
    else:
        dataset = fetch_ucirepo(id=id)  # Fetch dataset
        X, y = dataset.data.features.to_numpy(), dataset.data.targets.to_numpy()
        # Ensure the data folder exists
        os.makedirs(data_folder, exist_ok=True)
        np.save(X_file, X)
        np.save(y_file, y)

    return X, y


def loadgasdata():
    folder_path = r''  # Set as the directory of the dataset
    data_files = os.listdir(folder_path)
    data_files.sort()
    data_files_paths = [os.path.join(folder_path, file) for file in data_files]
    all_batches = []
    for file_path in data_files_paths:
        batch = pd.read_csv(file_path, sep=" ", header=None)
        all_batches.append(batch)
    combined_df = pd.concat(all_batches, ignore_index=True)
    y = combined_df.iloc[:, 0].to_numpy()
    X = combined_df.iloc[:, 1:].to_numpy()
    X = np.array([[float(value.split(":")[1]) for value in row[:-1]] for row in X])
    return X, y


def callsearch(iterations, task, num_classes, threads, estimate_size, param_distributions, scoring, metric, rand_state, classes=True):
    if classes:
        return RandomizedSearchCV(
        estimator=LightGBMRandomizedSearch(objective=task, num_class=num_classes, metric=metric, verbosity=-1,
                                           num_threads=threads, estimate_size=estimate_size),
        param_distributions=param_distributions,
        n_iter=iterations,
        cv=2,
        n_jobs=-1,
        random_state=rand_state,
        return_train_score=True,
        scoring=scoring,
        refit='memory_efficiency',
        verbose=4
        )
    else:
        return RandomizedSearchCV(
            estimator=LightGBMRandomizedSearch(objective=task, metric=metric,
                                               verbosity=-1,
                                               num_threads=threads, estimate_size=estimate_size),
            param_distributions=param_distributions,
            n_iter=iterations,
            cv=2,
            n_jobs=-1,
            random_state=rand_state,
            return_train_score=True,
            scoring=scoring,
            refit='memory_efficiency',
            verbose=4
        )


def run(dataset, iterations, rand_state, task="binary", train=True):
    if dataset == "magic":
        # Define dataset paths
        X, y = load_data("magic", 159)
    elif dataset == "spambase":
        X, y = load_data("spambase", 94)
    elif dataset == "statlog":
        num_classes = 6
        X, y = load_data("statlog", 146)
    elif dataset == "cycle":
        num_classes = 6
        X, y = load_data("cycle", 294)
    elif dataset == "concrete":
        X, y = load_data("concrete", 165)
    elif dataset == "superconductor":
        X, y = load_data("superconductor", 464)
    elif dataset == "gas":
        X, y = loadgasdata()
    else:
        raise ValueError(f"Dataset {dataset} not found. Cannot continue")

    y = y.ravel()
    if task != 'regression':
        le = LabelEncoder()
        y = le.fit_transform(y)
    X = MinMaxScaler().fit_transform(X)
    basepath = "/Users/ninaherrmann/docs/Lehre/BA/steen/CodeInstructions/"
    base_folder = f"{basepath}/data/{dataset}"  # Base directory to save/load results to/from
    if (train):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=rand_state)
        if task == 'multiclass':
            base_model, base_score, base_size, param_distributions, search_random = train_base_model(iterations, 'multiclass',
                                                                                                     fixed_hyperparameters,
                                                                                                     custom_hyperparameters,
                                                                                                     X_train, y_train,
                                                                                                     rand_state,
                                                                                                     num_class=num_classes,
                                                                                                     threads=threads,
                                                                                                     estimate=estimate_size)
        else:
            base_model, base_score, base_size, param_distributions, search_random = train_base_model(iterations, task,
                                                                                                     fixed_hyperparameters,
                                                                                                     custom_hyperparameters,
                                                                                                     X_train, y_train,
                                                                                                     rand_state,
                                                                                                     threads=threads,
                                                                                                     estimate=estimate_size)
        print(f"Base score: {base_score} \n Base size: {base_size}")

        if task == 'multiclass':
            scoring = {
                'memory_efficiency': CustomScorer(),
                'accuracy': 'accuracy',
                'memory': SizeMetric(),
                'accuracy_improvement': AccuracyImprovement(base_score),
                'memory_improvement': SizeImprovement(base_size),
                'thresholds': ThresholdMetric(),
                'trees': TreeMetric()
            }
            random_search = callsearch(iterations, 'multiclass', 6, threads, estimate_size, param_distributions, scoring, 'multi_logloss', rand_state)
        elif task == 'binary':
            scoring = {
                'memory_efficiency': CustomScorer(),
                'accuracy': 'accuracy',
                'memory': SizeMetric(),
                'accuracy_improvement': AccuracyImprovement(base_score),
                'memory_improvement': SizeImprovement(base_size),
                'thresholds': ThresholdMetric(),
                'trees': TreeMetric()
            }
            random_search = callsearch(iterations, 'binary', 1, threads, estimate_size, param_distributions, scoring, 'binary_logloss', rand_state)
        elif task == 'regression':
            scoring = {
                'memory_efficiency': CustomScorer(),
                'accuracy': 'neg_root_mean_squared_error',
                'memory': SizeMetric(),
                'accuracy_improvement': AccuracyImprovement(-base_score),
                'memory_improvement': SizeImprovement(base_size),
                'thresholds': ThresholdMetric(),
                'trees': TreeMetric()
            }
            random_search = callsearch(iterations, 'regression', 1, threads, estimate_size, param_distributions, scoring, 'rmse', rand_state, False)
        else :
            raise ValueError(f"task {task} not found. Cannot continue")

        random_search.fit(X_train, y_train)
        grid = {
            ### QUANTIZATION HYPERPARAMETERS ###
            "enable_quantization": [True],
            "quantization_type": ["affine", "scale"],
            "quantization_diff": ["leafs", "thresholds", "both"],
            "quantization_bits": [8, 16],
        }
        best_params = search_random.best_params_
        param_grid = {**grid}
        param_grid.update({k: [v] for k, v in best_params.items()})

        if task == 'binary':
            num_classes=1
            grid_search = GridSearchCV(
                estimator=LightGBMRandomizedSearch(objective='binary', num_class=1, metric='binary_logloss',
                                                   verbosity=-1,
                                                   num_threads=threads),
                param_grid=param_grid,
                cv=2,
                n_jobs=-1,
                return_train_score=True,
                scoring=scoring,
                refit='memory_efficiency',
                verbose=4
            )
        elif task == 'multiclass':
            grid_search = GridSearchCV(
                estimator=LightGBMRandomizedSearch(objective='multiclass', num_class=6, metric='multi_logloss',
                                                   verbosity=-1,
                                                   num_threads=threads),
                param_grid=param_grid,
                cv=2,
                n_jobs=-1,
                return_train_score=True,
                scoring=scoring,
                refit='memory_efficiency',
                verbose=4
            )
        elif task == 'regression':
            grid_search = GridSearchCV(
                estimator=LightGBMRandomizedSearch(objective='regression', num_class=1, metric='rmse', verbosity=-1,
                                                   num_threads=threads),
                param_grid=param_grid,
                cv=2,
                n_jobs=-1,
                return_train_score=True,
                scoring=scoring,
                refit='memory_efficiency',
                verbose=4
            )
        else:
            raise ValueError(f"task {task} not found. Cannot continue")

        grid_search.fit(X_train, y_train)
        folder = save_variables(f"i{iterations}_d{dataset}_t{task}",
            {"random_search": random_search, "search_random": search_random, "param_distributions": param_distributions,
             "random_search.cv_results_": random_search.cv_results_, "base_model": base_model,
             "grid_search": grid_search},
            base_folder)
    else:
        folder = "/Users/ninaherrmann/docs/Lehre/BA/steen/CodeInstructions/memory"  # Folder in the base directory to load results from
        loaded_vars = load_variables(folder)
        random_search = loaded_vars["random_search"]
        search_random = loaded_vars["search_random"]
        param_distributions = loaded_vars["param_distributions"]
        base_model = loaded_vars["base_model"]
        grid_search = loaded_vars["grid_search"]

    if train:
        best_models = evaluate(random_search.cv_results_, X_train, y_train, X_test, y_test, folder, rand_state, top=iterations, task=task, generate_outputs=False)


    analyzeResults(dataset, random_search.cv_results_, folder)
    analyze_grid_search(grid_search.cv_results_, folder)


def main():
    parser = argparse.ArgumentParser(description="Process some input arguments.")
    # Add --dataset, --type, and --train argument
    parser.add_argument('--iterations', type=int, required=True,
                        help='Number of iterations')
    parser.add_argument('--nrs', type=str, required=True,
                        help='Number of random states')
    parser.add_argument('--train', action='store_true', help='Should model be trained?')
    parser.add_argument('--no-train', action='store_false', dest='train', help='Do not train the model')
    parser.set_defaults(train=True)

    # Parse the command line arguments
    args = parser.parse_args()

    iterations = args.iterations
    train = args.train
    nrs = args.nrs

    datasets = ["magic", "spambase", "statlog", "cycle", "concrete", "superconductor"]
    for data in datasets:
        type = "binary"
        if data == "magic":
            type = "binary"
        if data == "spambase":
            type = "binary"
        if data == "statlog":
            type = "multiclass"
        if data == "cycle":
            type = "regression"
        if data == "concrete":
            type = "regression"
        if data == "superconductor":
            type = "regression"
        for i in range(int(nrs)):
            random.seed(i)
            random_state = random.getstate()
            random_number = random.randint(0, 4294967295)
            print(f"Random state {i}: {random_number}, Dataset: {data}, Type: {type}, Iterations: {iterations}, Train: {train}")
            run(data, iterations, random_number, task=type, train=train)


if __name__ == "__main__":
    main()
