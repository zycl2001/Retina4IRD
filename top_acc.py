import argparse
import ast
import os
import pandas as pd
import numpy as np
import yaml

n_classes = 17

def mean_and_95ci(metric_list):
    nums = len(metric_list)
    mean = np.mean(metric_list)
    std_error = np.std(metric_list) / np.sqrt(nums)
    ci_95 = 1.96 * std_error

    lower = np.around(mean - ci_95, 4)
    upper = np.around(mean + ci_95, 4)
    mean = np.around(mean, 4)

    return mean, (lower, upper)


def calculate_top_n_sensitivity_specificity(label_column,excel_file, n):
    df = pd.read_csv(excel_file)
    probability_columns = [str(i) for i in range(n_classes)]
    probabilities = df[probability_columns].to_numpy()
    true_labels = df[label_column].to_numpy()
    results = []

    total_TP = total_FN = total_TN = total_FP = 0
    for class_label in range(n_classes):
        top_n_predictions = probabilities.argsort(axis=1)[:, -n:][:, ::-1]
        tp = np.sum((true_labels == class_label) & (np.apply_along_axis(lambda x: class_label in x, axis=1, arr=top_n_predictions)))
        fn = np.sum((true_labels == class_label) & (~np.apply_along_axis(lambda x: class_label in x, axis=1, arr=top_n_predictions)))
        fp = np.sum((true_labels != class_label) & (np.apply_along_axis(lambda x: class_label in x, axis=1, arr=top_n_predictions)))
        tn = np.sum((true_labels != class_label) & (~np.apply_along_axis(lambda x: class_label in x, axis=1, arr=top_n_predictions)))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        total_TP += tp
        total_FN += fn
        total_FP += fp
        total_TN += tn

        results.append({'class_label': class_label, 'top_n': n, 'sensitivity': sensitivity, 'specificity': specificity})

    overall_sensitivity = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    overall_specificity = total_TN / (total_TN + total_FP) if (total_TN + total_FP) > 0 else 0

    return pd.DataFrame(results), overall_sensitivity, overall_specificity


def process_multiple_files(label_column,file_list, top_n_list, output_file):
    aggregated_results = []
    overall_metrics = []
    for n in top_n_list:
        overall_sensitivity_results = []
        overall_specificity_results = []
        sensitivity_results = {class_label: [] for class_label in range(n_classes)}
        specificity_results = {class_label: [] for class_label in range(n_classes)}

        for file in file_list:
            df,overall_sensitivity, overall_specificity = calculate_top_n_sensitivity_specificity(label_column,file, n)
            for class_label in range(n_classes):
                sensitivity_results[class_label].append(df.loc[df['class_label'] == class_label, 'sensitivity'].values[0])
                specificity_results[class_label].append(df.loc[df['class_label'] == class_label, 'specificity'].values[0])
            overall_sensitivity_results.append(overall_sensitivity)
            overall_specificity_results.append(overall_specificity)

        for class_label in range(n_classes):
            sensitivity_mean, sensitivity_ci = mean_and_95ci(sensitivity_results[class_label])
            specificity_mean, specificity_ci = mean_and_95ci(specificity_results[class_label])

            aggregated_results.append({
                'Top N': n,
                'Class ID': class_label,
                'Sensitivity': f"{sensitivity_mean} ({sensitivity_ci[0]}, {sensitivity_ci[1]})",
                'Specificity': f"{specificity_mean} ({specificity_ci[0]}, {specificity_ci[1]})"
            })
        overall_sensitivity_mean, overall_sensitivity_ci = mean_and_95ci(overall_sensitivity_results)
        overall_specificity_mean, overall_specificity_ci = mean_and_95ci(overall_specificity_results)
        overall_metrics.append({
            'Top N': n,
            'Overall Sensitivity': f"{overall_sensitivity_mean} ({overall_sensitivity_ci[0]}, {overall_sensitivity_ci[1]})",
            'Overall Specificity': f"{overall_specificity_mean} ({overall_specificity_ci[0]}, {overall_specificity_ci[1]})"
        })
    final_df = pd.DataFrame(aggregated_results)
    final_overall_metrics = pd.DataFrame(overall_metrics)
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        final_df.to_excel(writer, sheet_name='Sheet1', index=False)
        final_overall_metrics.to_excel(writer, sheet_name='Sheet2', index=False)
    return overall_metrics

def main_top_n(label_column,seed_count,cfp_res,oct_res):

    data_path = cfp_res.split('/cfp')[0]
    ensemble_res = f"{data_path}/results/ensemble/"

    data = {cfp_res: 'cfp',  oct_res: 'oct', ensemble_res: 'ensemble'}

    overall_metrics = []
    for file in data.keys():
        file_list = [f"{file}/seed_{i}/test_predict.csv" for i in range(seed_count)]
        top_n_list = [1, 2, 3, 4, 5]
        output_file = f"{data_path}/results/results_{data[file]}.xlsx"
        overall_metric=process_multiple_files(label_column,file_list, top_n_list, output_file)
        if data[file]=='ensemble':
            seed_path = f"{data_path}/results/seeds"
            process_multiple_files_individual_output(label_column,file_list, top_n_list, seed_path)

        overall_metrics.append(overall_metric)

def classification_ensemble_auroc(label_column,file_names, seed, num_teams=4):
    file_names = file_names[:num_teams]
    sum_df = None

    for file in file_names:
        df = pd.read_csv(file, usecols=[str(i) for i in range(n_classes)])
        if sum_df is None:
            sum_df = df
        else:
            sum_df += df

    average_probs = sum_df / len(file_names)
    predicted_categories = average_probs.idxmax(axis=1)
    output_df = average_probs.copy()
    output_df['predict'] = predicted_categories
    output_df['predict'] = output_df['predict'].astype(int)

    mutant_gene_label = pd.read_csv(file_names[0], usecols=[label_column])
    output_df[label_column] = mutant_gene_label[label_column]

    save_dir = f'{file_names[0].split("/cfp")[0]}/results/ensemble/seed_{seed}'
    os.makedirs(save_dir, exist_ok=True)
    output_df.to_csv(f"{save_dir}/test_predict.csv", index=False)



def generate_ensemble_auc(label_column,seed_count,cfp_res,oct_res):
    for i in range(seed_count):
        file_list = [f"{file}/seed_{i}/test_predict.csv" for file in [cfp_res,  oct_res]]
        classification_ensemble_auroc(label_column,file_list, i)

def process_multiple_files_individual_output(label_column,file_list, top_n_list, seeds_path):

    os.makedirs(seeds_path, exist_ok=True)
    count = 0
    for file in file_list:

        per_file_metrics_list = []
        per_file_overall_list = []

        for n in top_n_list:
            df, overall_sensitivity, overall_specificity = calculate_top_n_sensitivity_specificity(label_column,file, n)

            for class_label in range(n_classes):
                class_sens = df.loc[df['class_label'] == class_label, 'sensitivity'].values[0]
                class_spec = df.loc[df['class_label'] == class_label, 'specificity'].values[0]

                per_file_metrics_list.append({
                    'Top N': n,
                    'Class ID': class_label,
                    'Sensitivity': round(float(class_sens), 4),
                    'Specificity': round(float(class_spec), 4)
                })
            per_file_overall_list.append({
                'Top N': n,
                'Overall Sensitivity': round(float(overall_sensitivity), 4),
                'Overall Specificity': round(float(overall_specificity), 4)
            })
        per_file_metrics_df = pd.DataFrame(per_file_metrics_list)
        per_file_overall_df = pd.DataFrame(per_file_overall_list)
        output_path=f"{seeds_path}/results_seed{count}.xlsx"
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            per_file_metrics_df.to_excel(writer, sheet_name='Sheet1', index=False)
            per_file_overall_df.to_excel(writer, sheet_name='Sheet2', index=False)

        count+=1


def top_acc_acount(args):
    root_path=args.output_dir
    seed_count=args.seed_count
    label_column=args.label_column
    cfp_res = f"{root_path}/cfp/"
    oct_res = f"{root_path}/oct/"
    generate_ensemble_auc(label_column,seed_count, cfp_res, oct_res)
    main_top_n(label_column,seed_count, cfp_res, oct_res)

def get_args():
    parser = argparse.ArgumentParser(description='MultiMAE Finetune script', add_help=False)
    parser.add_argument('-c', '--config', default='cfgs/image_cls.yaml', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser.add_argument('--scale', type=str, default='None', help='New: (0.6, 1.0), old: None')
    args_config, remaining = parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)

    if args.scale == 'None':
        args.scale = None
    else:
        try:
            args.scale = ast.literal_eval(args.scale)
        except (ValueError, SyntaxError):
            raise ValueError("Invalid format for --scale. Must be a tuple or 'None'.")

    return args


if __name__ == '__main__':
    args=get_args()
    top_acc_acount(args)

