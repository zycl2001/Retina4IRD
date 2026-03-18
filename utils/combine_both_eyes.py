import os
import pandas as pd
import numpy as np
def entropy_quality_weighted_log_fusion_numpy(pL, pR, delta=1e-6, eps=0.05):
    K = pL.shape[1]

    # (1) 概率裁剪
    pL = np.clip(pL, delta, 1 - delta)
    pR = np.clip(pR, delta, 1 - delta)

    # (2) 预测熵
    HL = -np.sum(pL * np.log(pL), axis=1)
    HR = -np.sum(pR * np.log(pR), axis=1)

    # (3) 归一化不确定性
    logK = np.log(K)
    uL = HL / logK
    uR = HR / logK

    qL = np.maximum(1 - uL, eps)
    qR = np.maximum(1 - uR, eps)

    # (4) 权重
    wL = qL / (qL + qR)
    wR = qR / (qL + qR)

    # (5) log 概率融合
    s = wL[:, None] * np.log(pL) + wR[:, None] * np.log(pR)

    # softmax
    s_exp = np.exp(s - np.max(s, axis=1, keepdims=True))
    p_fused = s_exp / np.sum(s_exp, axis=1, keepdims=True)

    return p_fused, wL, wR


def combine_both_eyes(root_dir):
    # 概率列（字符串形式 0-16）
    prob_cols = [str(i) for i in range(17)]

    for subdir, dirs, files in os.walk(root_dir):
        if "train_predict.csv" in files or "only_test_predict_retfound.csv" in files:
            if "only_test_predict_retfound.csv" in files:
                file_path = os.path.join(subdir, "only_test_predict_retfound.csv")
            else:
                file_path = os.path.join(subdir, "train_predict.csv")
            # file_path = os.path.join(subdir, "traintest_predict.csv")
            print(f"Processing: {file_path}")

            df = pd.read_csv(file_path)
            result_rows = []

            grouped = df.groupby("Patients ID")

            for patient_id, group in grouped:

                if len(group) == 2:
                    # 两只眼
                    probs = group[prob_cols].values.astype(float)

                    # 分成左右眼
                    pL = probs[0:1]  # (1,17)
                    pR = probs[1:2]

                    fused_prob, wL, wR = entropy_quality_weighted_log_fusion_numpy(pL, pR)

                    # 用第一行作为模板
                    new_row = group.iloc[0].copy()
                    new_row[prob_cols] = fused_prob[0]

                    result_rows.append(new_row)

                else:
                    # 单眼直接保留
                    result_rows.append(group.iloc[0])

            result_df = pd.DataFrame(result_rows)

            save_path = os.path.join(subdir, "train_predict_combine_both_eyes.csv")
            result_df.to_csv(save_path, index=False)

            print(f"Saved to: {save_path}")