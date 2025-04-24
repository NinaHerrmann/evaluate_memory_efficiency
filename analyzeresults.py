import pandas as pd
from scipy import stats
from scipy.stats import linregress
from scipy.stats import randint, uniform
import numpy as np
def round_numbers(df):
    newdf = df
    numerical_cols = newdf.select_dtypes(include=['int64', 'float64']).columns
    newdf[numerical_cols] = newdf[numerical_cols].apply(lambda x: x.round(2))
    return newdf


def analyze_correlations(dataset, results_df, path):
    """
    Performs correlation analysis for continuous parameters and ANOVA tests for categorical parameters.
    Exports correlation analysis results to 'continuous.csv' and ANOVA test results to 'categorical.csv'.
    """
    metrics = ['accuracy', 'returnMemory', 'tradeoff']
    results_df['tradeoff'] = results_df['accuracy'] / results_df['returnMemory']
    categorical_params = [
        'enable_quantization', 'quantization_type', 'quantization_diff',
        'quantization_bits', "enable_pruning", "enable_merging",
        'enable_cegb', 'enable_prepruning',
    ]
    continuous_params = [
        'min_gain_to_split', 'cegb_penalty_split', 'min_data_in_leaf',
        'pruning_percentile', "num_clusters", 'max_depth', "num_leaves"
    ]
    correlation_df = pd.DataFrame(
        columns=['Parameter', 'Metric', 'SpearmanCorrelation', 'PearsonCorrelation', 'SpearmanCorrelation_p_value',
                 'PearsonCorrelation_p_value', 'Regression_Slope', 'Regression_Intercept', 'Regression_R2', ])
    anova_df = pd.DataFrame(columns=['Parameter', 'Metric', 'F_statistic', 'ANOVA_p_value'])
    for param in continuous_params:
        param_values = results_df[f'{param}']
        if param_values.empty:
            continue
        for metric in metrics:
            metric_values = results_df[metric]
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
        param_values = results_df[f'{param}'].dropna()
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
                    'ANOVA_p_value': [f"{round(p_value,3)}" if pd.notna(p_value) else np.nan]
                })

                if not new_data.isna().all(axis=None) and not new_data.empty:
                    anova_df = pd.concat([anova_df, new_data], ignore_index=True)
            except ValueError as e:
                print(f"Could not perform ANOVA for {param} on {metric}: {str(e)}")

    roundeddf = round_numbers(correlation_df)
    latex_table = roundeddf.to_latex(index=False)
    with open(f'{path}/{dataset}table_continous_singleview.tex', 'w') as f:
         f.write(latex_table)
    roundanova = round_numbers(anova_df)
    latex_table = roundanova.to_latex(index=False)
    with open(f'{path}/{dataset}_table_anovacat_singleview.tex', 'w') as f:
         f.write(latex_table)

def anova_over_datasets(results_df):
    metrics = ['accuracy', 'returnMemory', 'tradeoff']
    results_df['tradeoff'] = results_df['accuracy'] / results_df['returnMemory']

    anova_df = pd.DataFrame(columns=['Parameter', 'Metric', 'F_statistic', 'ANOVA_p_value'])

    bool_cols = results_df.select_dtypes(include=[bool]).columns
    bool_and_metrics_cols = results_df.select_dtypes(include=[bool]).columns.tolist() + metrics
    melted_df = pd.melt(results_df[bool_and_metrics_cols], id_vars=metrics, var_name='feature', value_name='value')
    filtered_df = melted_df[(melted_df['value'] == True) | (~melted_df['feature'].isin(bool_cols))]
    result_df = filtered_df.drop(columns=['value'])
    param_values = result_df['feature'].dropna()
    for metric in metrics:
        print(result_df)
        metric_values = result_df.loc[param_values.index, metric]

        groups = [metric_values[param_values == value] for value in param_values.unique()]
        try:
            f_statistic, p_value = stats.f_oneway(*groups)

            new_data = pd.DataFrame({
                'Parameter': ['Features'],
                'Metric': [metric],
                'F_statistic': [round(f_statistic, 3) if pd.notna(f_statistic) else np.nan],
                'ANOVA_p_value': [f"{round(p_value, 4)}" if pd.notna(p_value) else np.nan]
            })

            if not new_data.isna().all(axis=None) and not new_data.empty:
                anova_df = pd.concat([anova_df, new_data], ignore_index=True)
        except ValueError as e:
            print(f"Could not perform ANOVA for Features on {metric}: {str(e)}")
    return anova_df

def anova_for_methods(df):
    pruned = df[(df['enable_pruning'] == True) & (df['enable_prepruning'] == False) & (df['enable_cegb'] == False) & (
                df['enable_merging'] == False) & (df['enable_quantization'] == False)]
    pre_pruned = df[
        (df['enable_prepruning'] == True) & (df['enable_pruning'] == False) & (df['enable_cegb'] == False) & (
                    df['enable_merging'] == False) & (df['enable_quantization'] == False)]
    cegb = df[(df['enable_cegb'] == True) & (df['enable_pruning'] == False) & (df['enable_prepruning'] == False) & (
                df['enable_merging'] == False) & (df['enable_quantization'] == False)]
    quant8 = df[(df['quantization_bits'] == 8) & (df['enable_quantization'] == True)]
    quant16 = df[(df['quantization_bits'] == 16) & (df['enable_quantization'] == True)]
    merging = df[
        (df['enable_merging'] == True) & (df['enable_pruning'] == False) & (df['enable_prepruning'] == False) & (
                    df['enable_cegb'] == False) & (df['enable_quantization'] == False)]
    frames = [pruned, cegb, pre_pruned, quant16, quant8, merging]
    result = pd.concat(frames)
    #analyze_correlations(dataset, result, path)
    return anova_over_datasets(result)



datasets = ['concrete', 'cycle', 'magic', 'spambase', 'statlog', 'superconductor']
anova_df = pd.DataFrame(columns=['Parameter', 'Metric', 'F_statistic', 'ANOVA_p_value'])

for dataset in datasets:
    path = 'data/' + dataset + '/202504_i1000/'
    df = pd.read_csv(path + 'categorical.csv', sep=',', index_col=False)
    dffull = pd.read_csv(path + 'print.csv', sep=';', index_col=False)
    roundanova = round_numbers(df)
    newanova_df = anova_for_methods(dffull)
    if anova_df is not None:
        anova_df = pd.concat([anova_df, newanova_df])
    else:
        anova_df = newanova_df
    # latex_table = roundanova.to_latex(index=False)
    # with open(f'{path}/{dataset}_table_anovacat.tex', 'w') as f:
    #     f.write(latex_table)


print(round_numbers(anova_df).to_latex(index=False))