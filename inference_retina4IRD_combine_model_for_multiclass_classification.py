import argparse
import ast


import pandas as pd
import pickle
import os

import yaml

from sklearn.preprocessing import StandardScaler


import warnings
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
model_type = [['age', 'sex', 'continue_time', 'sick_age', 'genetics', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10','11','12','13','14','15','16']]


def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_test_data(meta_data_dir,index):
    test_file = os.path.join(meta_data_dir, 'test_predict.csv')
    if not os.path.exists(test_file):
        print(f"File does not exist: {test_file}")
        return None, None

    try:
        if test_file.endswith('.csv'):
            test_data = pd.read_csv(test_file).copy()
        else:
            test_data = pd.read_excel(test_file,engine='openpyxl').copy()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None

    try:
        test_input = test_data[model_type[index]]
        test_label = test_data['gene_label'].to_numpy().ravel()
    except KeyError as e:
        print(f"Column name error: {e}")
        return None, None

    return test_input, test_label

def test_model(args,model_path, meta_data_dir,index=0):
    model = load_model(model_path)
    if model is None:
        return

    test_input, test_label = load_test_data(meta_data_dir,index)
    if test_input is None or test_label is None:
        return

    if '0' in test_input.columns:
        columns_to_normalize = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                                '16','age', 'continue_time', 'sick_age']
        scaler = StandardScaler()
        scaler.fit_transform(test_input[columns_to_normalize])
        train_input_scaled = scaler.transform(test_input[columns_to_normalize])
        test_input[columns_to_normalize] = pd.DataFrame(train_input_scaled, columns=columns_to_normalize)

    test_preds = model.predict_proba(test_input)

    test_preds_df = pd.DataFrame(test_preds, columns=[str(i) for i in range(17)])
    test_preds_df['gene_label'] = test_label
    test_preds_df.to_csv(os.path.join(args.output_dir, 'test_predict.csv'), index=False)

def run_one_test(args):
    if args.in_domains == "rgb":
        model_path=args.combine_weight_cfp
        args.output_dir = os.path.join(args.output_dir, 'cfp')
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        model_path=args.combine_weight_oct
        args.output_dir = os.path.join(args.output_dir, 'oct')
        os.makedirs(args.output_dir, exist_ok=True)

    test_model(args,model_path,args.csv_path,index=0)

def get_args():
    parser = argparse.ArgumentParser(description='MultiMAE Finetune script', add_help=False)

    parser.add_argument('--test', action='store_true')
    parser.add_argument('--in_domains', default='rgb')
    parser.add_argument('--csv_path', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--label_column', type=str, default='')
    parser.add_argument('--combine_weight_cfp', type=str, default='')
    parser.add_argument('--combine_weight_oct', type=str, default='')
    parser.add_argument('-c', '--config', default='cfgs/combine_cls.yaml', type=str)
    parser.add_argument('--scale', type=str, default='None')

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)

        for k, v in cfg.items():
            if hasattr(args, k):  # 防止yaml多余字段
                if getattr(args, k) == parser.get_default(k):
                    setattr(args, k, v)

    if args.scale == 'None':
        args.scale = None
    else:
        args.scale = ast.literal_eval(args.scale)

    print(args)
    return args

if __name__ == '__main__':
    args = get_args()
    run_one_test(args)