import argparse
import ast
import shutil

import pandas as pd
import pickle
import os

import yaml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import warnings

from joblib import dump
warnings.filterwarnings("ignore")

model_type = [['age', 'sex', 'continue_time', 'sick_age', 'genetics', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10','11','12','13','14','15','16']]
predict_files = ["test_predict.csv", "train_predict.csv", "val_predict.csv"]
folders = ["cfp", "oct"]

def jiaQuan2(seed_count,label_column,folders,meta_data_path, result_data_path,weight, output_path,jiaquan_count_output_path):
    meta_files=predict_files
    for folder in folders:
        for se in range(seed_count):
            seed=f"seed_{se}"
            for i in range(len(meta_files)):

                meta_file = os.path.join(meta_data_path, folder, seed, meta_files[i])
                result_file = os.path.join(result_data_path, folder, seed, predict_files[i])


                if meta_file.endswith('.csv'):
                    meta_data = pd.read_csv(meta_file)
                else:
                    meta_data = pd.read_excel(meta_file,engine='openpyxl')

                if result_file.endswith('.csv'):
                    result_data = pd.read_csv(result_file)
                else:
                    result_data = pd.read_excel(result_file,engine='openpyxl')

                weight_s1 = weight
                weight_s2 = 1 - weight_s1

                weighted_data = weight_s1 * meta_data[[str(i) for i in range(17)]] + \
                                weight_s2 * result_data[[str(i) for i in range(17)]]

                weighted_data[label_column] = meta_data[label_column]

                output_dir = os.path.join(output_path, folder, seed)
                output_dir_count=os.path.join(jiaquan_count_output_path, folder, seed)
                os.makedirs(output_dir, exist_ok=True)
                os.makedirs(output_dir_count, exist_ok=True)


                output_file = os.path.join(output_dir, predict_files[i])
                output_file_count=os.path.join(output_dir_count, predict_files[i])
                if output_file.endswith('.csv'):
                    weighted_data.to_csv(output_file, index=False)
                    weighted_data.to_csv(output_file_count, index=False)
                else:
                    weighted_data.to_excel(output_file, index=False)
                    weighted_data.to_excel(output_file_count, index=False)



def combination_meta_predict_score_one_time(seed_count,folders,meta_data_dir, result_dir, output_dir):
    meta_files = predict_files
    os.makedirs(output_dir, exist_ok=True)

    for folder in folders:
        meta_folder_path = os.path.join(meta_data_dir, folder)
        result_folder_path = os.path.join(result_dir, folder)
        output_folder_path = os.path.join(output_dir, folder)

        os.makedirs(output_folder_path, exist_ok=True)

        for se in range(seed_count):
            seed = f"seed_{se}"
            meta_seed_path = os.path.join(meta_folder_path, seed)
            result_seed_path = os.path.join(result_folder_path, seed)
            output_seed_path = os.path.join(output_folder_path, seed)

            os.makedirs(output_seed_path, exist_ok=True)


            for i in range(len(meta_files)):
                meta_file_path = os.path.join(meta_seed_path, meta_files[i])
                result_file_path = os.path.join(result_seed_path, predict_files[i])
                output_file_path = os.path.join(output_seed_path, meta_files[i])

                if os.path.exists(meta_file_path) and os.path.exists(result_file_path):

                    if meta_file_path.endswith('.csv'):
                        meta_data = pd.read_csv(meta_file_path)
                    else:
                        meta_data = pd.read_excel(meta_file_path,engine='openpyxl')

                    columns_to_extract = [
                        "age", "sex", "continue_time", "sick_age",
                        "genetics"
                    ]
                    meta_data_extracted = meta_data[columns_to_extract]

                    if result_file_path.endswith('.csv'):
                        result_data = pd.read_csv(result_file_path)
                    else:
                        result_data = pd.read_excel(result_file_path,engine='openpyxl')

                    combined_data = pd.concat([meta_data_extracted, result_data], axis=1)

                    if output_file_path.endswith('.csv'):
                        combined_data.to_csv(output_file_path, index=False)
                    else:
                        combined_data.to_excel(output_file_path, index=False)
                else:
                    print(f"File missing, skipping: {meta_file_path} or {result_file_path}")

def pre_score(label_column,model_index, meta_data_dir, output_dir):
    meta_files = predict_files
    if model_index >= len(model_type):
        print(f"Error: model_index {model_index} is out of range. Available indices are 0 to {len(model_type) - 1}.")
        return

    train_file = os.path.join(meta_data_dir, meta_files[1])
    test_file = os.path.join(meta_data_dir, meta_files[0])
    val_file = os.path.join(meta_data_dir, meta_files[2])

    for file_path in [train_file, test_file, val_file]:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
    try:
        train_data = pd.read_csv(train_file).copy()
        test_data = pd.read_csv(test_file).copy()
        val_data = pd.read_csv(val_file).copy()
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    try:
        train_input = train_data[model_type[model_index]]
        train_label = train_data[label_column].to_numpy().ravel()

        test_input = test_data[model_type[model_index]]
        test_label = test_data[label_column].to_numpy().ravel()

        val_input = val_data[model_type[model_index]]
        val_label = val_data[label_column].to_numpy().ravel()
    except KeyError as e:
        print(f"Invalid column name in the file: {e}")
        return
    if '0' in train_input.columns:
        columns_to_normalize = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15','16']
        scaler = StandardScaler()
        train_input_scaled = scaler.fit_transform(train_input[columns_to_normalize])
        test_input_scaled = scaler.transform(test_input[columns_to_normalize])
        val_input_scaled = scaler.transform(val_input[columns_to_normalize])

        train_input[columns_to_normalize] = pd.DataFrame(train_input_scaled, columns=columns_to_normalize)
        test_input[columns_to_normalize] = pd.DataFrame(test_input_scaled, columns=columns_to_normalize)
        val_input[columns_to_normalize] = pd.DataFrame(val_input_scaled, columns=columns_to_normalize)

        relative_path = '/'.join(meta_data_dir.split('/')[-2:])

        scaler_set_path = os.path.join(output_dir, relative_path)

        if not os.path.exists(scaler_set_path):
            os.makedirs(scaler_set_path)

        scaler_path = os.path.join(scaler_set_path, 'scaler.joblib')
        dump(scaler, scaler_path)


    l0_models = [
        ("XGBoost", XGBClassifier(
            booster='dart',
            n_estimators=21,
            reg_lambda=0.46,
        )),
        ("LightGBM", LGBMClassifier(
            num_iterations=33,
            num_leaves=8,
            lambda_l1=0.21,
            verbosity=-1,
        )),
        ("KNN", KNeighborsClassifier(
            n_neighbors=9,
        )),
    ]
    l1_model = MLPClassifier(
        hidden_layer_sizes=100,
        random_state=177,
        learning_rate_init=0.0186,
    )

    num_folds = 3
    random_state = 100
    model = StackingClassifier(estimators=l0_models, final_estimator=l1_model)
    cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    best_score = -1
    best_model=None
    for train_index, valid_index in cv.split(train_input, train_label):
        X_train, X_valid = train_input.iloc[train_index], train_input.iloc[valid_index]
        y_train, y_valid = train_label[train_index], train_label[valid_index]

        model.fit(X_train, y_train)
        valid_preds = model.predict(X_valid)
        acc = accuracy_score(y_valid, valid_preds)

        if acc > best_score:
            best_score = acc
            print(f'best Accuracy:{best_score}')
            best_model = pickle.loads(pickle.dumps(model))

    test_preds = best_model.predict_proba(test_input)
    val_preds = best_model.predict_proba(val_input)
    train_preds = best_model.predict_proba(train_input)
    print("Start saving prediction results...")

    relative_path = '/'.join(meta_data_dir.split('/')[-2:])
    output_dir = os.path.join(output_dir, relative_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_preds_df = pd.DataFrame(train_preds, columns=[str(i) for i in range(17)])
    train_preds_df[label_column] = train_label[:len(train_preds)]
    train_preds_df.to_csv(os.path.join(output_dir, 'train_predict.csv'), index=False)

    val_preds_df = pd.DataFrame(val_preds, columns=[str(i) for i in range(17)])
    val_preds_df[label_column] = val_label
    val_preds_df.to_csv(os.path.join(output_dir, 'val_predict.csv'), index=False)
    test_preds_df = pd.DataFrame(test_preds, columns=[str(i) for i in range(17)])
    test_preds_df[label_column] = test_label
    test_preds_df.to_csv(os.path.join(output_dir, 'test_predict.csv'), index=False)

    print("Prediction results have been saved to the specified path")

    model_save_path = os.path.join(output_dir, 'stacking_model.pkl')
    with open(model_save_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Model has been saved to {model_save_path}")

def run_one_time(args):
    root_dir = args.csv_path
    output_dir = args.output_dir
    count = args.count
    weight = args.weight
    seed_count = args.count
    label_column = args.label_column
    jiaquan_output_path = f"{output_dir}/jiaquan/"
    output_combination_dir = f"{output_dir}/test/"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(jiaquan_output_path, exist_ok=True)
    os.makedirs(output_combination_dir, exist_ok=True)
    model_type_index = 0

    for i in range(0, count + 1):
        jiaquan_count_output_path = fr"{jiaquan_output_path}/{i}_jiaquan_Tianxin"
        if not os.path.exists(jiaquan_count_output_path):
            os.makedirs(jiaquan_count_output_path)

        for folder in folders:
            folder_path = os.path.join(root_dir, folder)
            print(folder)
            for se in range(seed_count):
                seed = fr"seed_{se}"
                print(seed)
                seed_path = os.path.join(folder_path, seed)
                pre_score(label_column, model_type_index, seed_path, output_dir)
        if i == 0:
            jiaQuan2(seed_count, label_column, folders, root_dir, output_dir, weight, jiaquan_output_path,
                     jiaquan_count_output_path)
            combination_meta_predict_score_one_time(seed_count, folders, root_dir, jiaquan_output_path,
                                                    output_combination_dir)
            root_dir = output_combination_dir
        else:
            jiaQuan2(seed_count, label_column, folders, output_combination_dir, output_dir, weight, jiaquan_output_path,
                     jiaquan_count_output_path)
            combination_meta_predict_score_one_time(seed_count, folders, root_dir, jiaquan_output_path,
                                                    output_combination_dir)
            root_dir = output_combination_dir

        if os.path.exists(jiaquan_count_output_path):
            shutil.rmtree(jiaquan_count_output_path)

    if os.path.exists(output_combination_dir):
        shutil.rmtree(output_combination_dir)

    if os.path.exists(jiaquan_output_path):
        shutil.rmtree(jiaquan_output_path)

def get_args():
    parser = argparse.ArgumentParser(description='MultiMAE Finetune script', add_help=False)
    parser.add_argument('-c', '--config', default='cfgs/combine_cls.yaml', type=str, metavar='FILE',
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
    print(args)
    return args


if __name__ == '__main__':

    args = get_args()
    run_one_time(args)
