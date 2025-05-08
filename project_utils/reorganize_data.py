import os
import shutil

def reorganize_data(source_root, target_root, eval_root):
    # 定义所有模态
    modalities = ['ADC', 'CP', 'DWI', 'EP', 'NP', 'T2WI', 'T1WI']
    
    # 定义数据集类型
    dataset_types = ['train']# , 'val']

    for dataset_type in dataset_types:
        classify_folder = 'classify_train' if dataset_type == 'train' else 'classify_test'
        eval_dataset_type = 'train' if dataset_type == 'train' else 'test'
    
        for modality in modalities:
            img_path = os.path.join(source_root, modality, dataset_type, 'image')
            pred_path = os.path.join(eval_root, f'seg_results_{eval_dataset_type}_{modality}')

            if not os.path.exists(img_path):
                print(f"Warning: Path not found for modality {modality} in {dataset_type}")
                continue

            # 获取预测标签文件名列表
            pred_filenames = [filename for filename in os.listdir(pred_path) if filename.endswith('.nii.gz')]
        
            for pred_filename in pred_filenames:
                # 获取病例名
                case_name = os.path.splitext(pred_filename)[0].rsplit('_predict', 1)[0]
            
                # 查找原始图像文件
                img_filename = f"{case_name}.nii.gz"
                src_img = os.path.join(img_path, img_filename)
                if not os.path.exists(src_img):
                    print(f"Warning: Image not found for {img_filename} in {modality} ({dataset_type})")
                    continue

                # 创建新的目录结构
                new_img_dir = os.path.join(target_root, classify_folder, case_name, modality, 'image')
                new_label_dir = os.path.join(target_root, classify_folder, case_name, modality, 'label')
                os.makedirs(new_img_dir, exist_ok=True)
                os.makedirs(new_label_dir, exist_ok=True)

                # 复制图像文件
                dst_img = os.path.join(new_img_dir, img_filename)
                shutil.copy2(src_img, dst_img)

                # 复制预测标签文件
                src_pred = os.path.join(pred_path, pred_filename)
                dst_pred = os.path.join(new_label_dir, pred_filename)
                shutil.copy2(src_pred, dst_pred)

    print("Data reorganization completed.")

# 使用示例
source_root = '../data'  # 原始数据的根目录
target_root = '../data'  # 要存放重组后数据的目录
eval_root = '../eval'    # 预测标签所在的目录

reorganize_data(source_root, target_root, eval_root)