# 在这个修改的版本中，我添加了一个 try-except 块来捕获异常。
# 如果特征提取失败，病例名、模态和图像文件名将被记录到 skipped_cases 列表中。然后，这些信息将被写入到一个文件中
# 添加了一个检查来确保 img_path 存在。如果 img_path 不存在，则会打印一个警告消息并跳过该模态。
import sys
sys.path.append('..')

import os
import SimpleITK
import pandas as pd
from radiomics import featureextractor
import numpy as np
import random

def extract_features(case_path, params_path):
    extractor = featureextractor.RadiomicsFeatureExtractor(params_path)
    modalities = ['ADC','CP', 'DWI', 'EP', 'NP', 'T1WI', 'T2WI']
    
    case_features = {}

    skipped_cases = []

    for i, modality in enumerate(modalities, start=1):
        img_path = os.path.join(case_path, modality, 'image')
        mask_path = os.path.join(case_path, modality, 'label')
        
        print(f"Processing modality: {modality}")
        print(f"img_path: {img_path}")
        
        if not os.path.exists(img_path):
            print(f"Warning: Modality {modality} not found for case {os.path.basename(case_path)}")
            continue
        
        for img_file in sorted(os.listdir(img_path)):
            if img_file.endswith('.nii.gz'):
                img_full_path = os.path.join(img_path, img_file)
                mask_file = f"{img_file.split('.')[0]}_predict.nii.gz"
                mask_full_path = os.path.join(mask_path, mask_file)
                
                print(f"Processing image: {img_file}")
                print(f"Processing mask: {mask_file}")
                
                if not os.path.exists(mask_full_path):
                    print(f"Warning: Mask not found for {img_file} in {modality}")
                    continue
                
                img = SimpleITK.ReadImage(img_full_path)
                mask = SimpleITK.ReadImage(mask_full_path)
                
                try:
                    featureVector = pd.Series(extractor.execute(img, mask))
                    featureVector = featureVector.add_suffix(f"_{i}")
                    case_features.update(featureVector)
                except Exception as e:
                    print(f"Error extracting features for {img_file} in {modality}: {str(e)}")
                    case_name = os.path.basename(case_path)
                    skipped_cases.append((case_name, modality, img_file))
    
    return case_features, skipped_cases

def main():
    # 固定随机种子
    np.random.seed(42)
    random.seed(42)

    data_root = "../data"
    params_path = "../config/Params_myself1.yaml"
    # output_path = "../data/features/train_features.csv"
    output_path = "../data/features/test_features.csv"
    skipped_cases_path = "../data/skipped_cases.txt"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_features = pd.DataFrame()
    # train_path = os.path.join(data_root, 'classify_train')  # 修改为新的训练数据路径
    test_path = os.path.join(data_root, 'classify_test')  # 修改为新的训练数据路径    
    skipped_cases = []
    for case in sorted(os.listdir(test_path)):
        case_path = os.path.join(test_path, case)
        print(f"Processing case: {case}")
        case_features, case_skipped_cases = extract_features(case_path, params_path)
        skipped_cases.extend(case_skipped_cases)
        if case_features:
            case_series = pd.Series(case_features, name=case)
            all_features = all_features.join(case_series, how='outer')
    
    all_features = all_features.T
    
    drop_dp = all_features.filter(regex=('diagnostics.*'))
    all_features = all_features.drop(drop_dp.columns, axis=1)
    
    all_features.reset_index(inplace=True)
    all_features.rename(columns={'index': ''}, inplace=True)
    
    all_features.to_csv(output_path, index=False)
    print(f"Features for training set saved to {output_path}")
    
    with open(skipped_cases_path, 'w') as f:
        for case in skipped_cases:
            f.write(f"{case[0]} - {case[1]} - {case[2]}\n")
    print(f"Skipped cases saved to {skipped_cases_path}")

if __name__ == "__main__":
    main()