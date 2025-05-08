# 该代码想比较于evaluate_model_1.py多了以下内容：
# 由于模型的分割性能较低，所以对一些数据进行前向推理预测时，预测结果为0
# 因此，针对这些数据，改代码进行记录，并以txt文件形式保存预测结果为0的数据名
# 即使分割结果为空（全零），我们仍然创建并保存一个全零的标签文件。这确保了每个病例都有对应的标签文件，即使是空的。

import sys
sys.path.append('..')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction

from config.setting import get_config
from project_utils.model_creater import create_model
from project_utils.data_creater import create_data
from trainer.seg_trainer import dice_acc

def get_arguments():
    parser = argparse.ArgumentParser(description='Segmentation Evaluation')
    parser.add_argument('--cfg', type=str, 
                        # default='../config/ADC_infer.yaml'
                        # default='../config/CP_infer.yaml'
                        # default='../config/DWI_infer.yaml'
                        # default='../config/EP_infer.yaml'
                        # default='../config/NP_infer.yaml'
                        # default='../config/T1WI_infer.yaml'
                        default='../config/T2WI_infer.yaml'
                        )
    parser.add_argument('--cuda', action='store_true', default=True)
    args = parser.parse_args()
    config = get_config(args)
    return args, config

def load_model(args, config, checkpoint_path):
    model, _ = create_model(args, config)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if args.cuda:
        model = model.cuda()
    model.eval()
    return model

def evaluate_model(model, data_loader, config):
    seg_total = 0
    seg_correct = 0
    dice_total = 0.0
    individual_dice_scores = []
    case_names = []

    post_label = AsDiscrete(to_onehot=config.MODEL.NUM_CLASSES)
    post_pred = AsDiscrete(argmax=True, to_onehot=config.MODEL.NUM_CLASSES)

    dice_acc.reset()

    with torch.no_grad():
        for idx, (image, seg_lab, image_name, seg_lab_name) in enumerate(data_loader):
            data, seg_target = image.cuda(), seg_lab.cuda()
            
            seg_logits = model(data)
            
            # Segmentation accuracy
            _, predicted = seg_logits.max(1)
            seg_correct += predicted.eq(seg_target).sum().item() / seg_target.numel()
            seg_total += 1

            # Calculate individual Dice scores
            val_labels_list = decollate_batch(seg_target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(seg_logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            
            # Calculate Dice score for each sample
            for i in range(len(val_labels_convert)):
                sample_dice = DiceMetric(include_background=True, reduction=MetricReduction.MEAN)
                sample_dice(y_pred=[val_output_convert[i]], y=[val_labels_convert[i]])
                dice_value = sample_dice.aggregate().item()
                individual_dice_scores.append(dice_value)
                
                # Extract case name from the full path
                base_name = os.path.basename(image_name[i])
                case_name = base_name.split('.')[0]
                case_names.append(case_name)

            # Overall Dice score
            dice_acc(y_pred=val_output_convert, y=val_labels_convert)
            seg_acc = dice_acc.aggregate()
            dice_total += seg_acc[0].item()

    seg_accuracy = seg_correct / seg_total
    dice_score = dice_total / len(data_loader)

    return seg_accuracy, dice_score, case_names, individual_dice_scores

def visualize_segmentation_results(model, data_loader, config, sequence_name):
    model.eval()
    post_pred = AsDiscrete(argmax=True, to_onehot=config.MODEL.NUM_CLASSES)

    # output_dir = f'./seg_results_train_{sequence_name}'    # 训练集与验证集的保存路径
    output_dir = f'./seg_results_test_{sequence_name}'    # 测试集保存路径
    os.makedirs(output_dir, exist_ok=True)

    empty_cases = []

    with torch.no_grad():
        for idx, (image, seg_lab, image_name, seg_lab_name) in enumerate(data_loader):
            data = image.cuda()
            seg_logits = model(data)
            
            seg_outputs = torch.softmax(seg_logits, dim=1)
            seg_outputs = seg_outputs.cpu().numpy()

            # 处理每个样本
            for i in range(seg_outputs.shape[0]):
                result = np.argmax(seg_outputs[i], axis=0)

                # 检查预测结果是否为空白的
                if np.sum(result) == 0:
                    print(f"预测结果为空的病例名：{image_name[i]}")
                    empty_cases.append(image_name[i])
                    # 即使结果为空，也创建一个全零的标签
                    result = np.zeros_like(result)

                # 将结果转换为 uint8 类型
                result = result.astype(np.uint8)
                
                # 创建NIfTI图像
                new_seg = nib.Nifti1Image(result, np.eye(4))
                
                # 修改文件名格式
                base_name = os.path.basename(image_name[i])
                case_name = base_name.split('.')[0]  # 获取病例名（不包括文件扩展名）
                
                # 保存结果
                save_path = os.path.join(output_dir, f"{case_name}_predict.nii.gz")
                nib.save(new_seg, save_path)

    # 保存空白病例列表
    with open(f"{sequence_name}_empty_cases_test.txt", "w") as f:
        for case in empty_cases:
            f.write(case + "\n")

    print(f"Total empty cases for {sequence_name}: {len(empty_cases)}")

def save_results(results, save_path):
    with open(save_path, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

def save_dice_scores_to_excel(case_names, dice_scores, sequence_name):
    try:
        # Try to read existing Excel file
        df_existing = pd.read_excel('test_evaluation_results.xlsx', sheet_name=None)
    except FileNotFoundError:
        df_existing = {}

    # Create new DataFrame with case names and Dice scores
    df_new = pd.DataFrame({
        'Case Name': case_names,
        f'{sequence_name} Dice Score': dice_scores
    })

    # If sheet1 exists, update it; if not, create it
    if 'sheet1' in df_existing:
        df_sheet1 = df_existing['Sheet1']
        # If this sequence already has a column, update it
        if f'{sequence_name} Dice Score' in df_sheet1.columns:
            # Update existing values and add new ones
            df_sheet1 = df_sheet1.set_index('Case Name')
            df_new = df_new.set_index('Case Name')
            df_sheet1[f'{sequence_name} Dice Score'] = df_new[f'{sequence_name} Dice Score']
            df_sheet1 = df_sheet1.reset_index()
        else:
            # Add new column
            df_sheet1 = df_sheet1.merge(df_new, on='Case Name', how='outer')
    else:
        df_sheet1 = df_new

    # Create ExcelWriter object
    with pd.ExcelWriter('evaluation_results.xlsx', engine='openpyxl', mode='w' if not df_existing else 'a') as writer:
        # Write all existing sheets except sheet1
        for sheet_name, df in df_existing.items():
            if sheet_name != 'sheet1':
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Write the updated sheet1
        df_sheet1.to_excel(writer, sheet_name='sheet1', index=False)

def main():
    args, config = get_arguments()
    sequence_name = os.path.basename(args.cfg).split('_')[0]
    
    checkpoint_path = config.TEST.MODEL_PATH
    model = load_model(args, config, checkpoint_path)
    
    # train_generator, _, _ = create_data(args, config)  
    # _, val_generator, _ = create_data(args, config)  
    _, _, test_generator = create_data(args, config)  

    # 评估模型并获取单个样本的Dice分数
    seg_accuracy, dice_score, case_names, individual_dice_scores = evaluate_model(model, test_generator, config)
    
    # 保存单个样本的Dice分数到Excel
    save_dice_scores_to_excel(case_names, individual_dice_scores, sequence_name)
    
    # 可视化分割结果
    visualize_segmentation_results(model, test_generator, config, sequence_name)

    # 保存整体结果
    results = {
        "Sequence": sequence_name,
        "Segmentation Accuracy": seg_accuracy,
        "Average Dice Score": dice_score
    }
    print(f"Segmentation Accuracy: {seg_accuracy}")
    print(f"Average Dice Score: {dice_score}")
    
    save_results(results, f"evaluation_results_{sequence_name}.txt")

    print(f"Evaluation for {sequence_name} completed. Results saved.")

if __name__ == "__main__":
    main()