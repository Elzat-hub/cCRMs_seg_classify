import os
import shutil
import pandas as pd


def move_files_not_in_csv(source_folder, csv_file, target_folder):
    # 读取 CSV 文件，提取 NIfTI 文件名列表
    df = pd.read_csv(csv_file)
    nii_files_in_csv = df['filename'].tolist()

    # 遍历源文件夹中的所有文件
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.nii.gz'):  # 假设文件是 NIfTI 文件
                source_file_path = os.path.join(root, file)
                if file not in nii_files_in_csv:  # 检查文件是否在 CSV 中
                    # 移动文件到目标文件夹
                    target_file_path = os.path.join(target_folder, file)
                    shutil.move(source_file_path, target_file_path)
                    print(f"Moved {file} to {target_folder}")


def count_niigz_files(folder_path):
    """
    统计文件夹下的NIfTI-GZ文件（.nii.gz文件）数量。

    参数:
    - folder_path (str): 文件夹路径。

    返回值:
    - count (int): NIfTI-GZ文件数量。
    """
    count = 0
    for file in os.listdir(folder_path):
        if file.endswith('.nii.gz'):
            count += 1
    return count


def find_missing_niigz(csv_file, folder_path):
    """
    在当前文件夹中查找 CSV 文件中存在但当前文件夹下不存在的NIfTI-GZ文件。

    参数:
    - csv_file (str): CSV 文件路径。
    - folder_path (str): 当前文件夹路径。

    返回值:
    - missing_files (list): CSV 文件中存在但当前文件夹下不存在的文件列表。
    """
    # 读取 CSV 文件，提取文件名列表
    df = pd.read_csv(csv_file)
    csv_files = df['filename'].tolist()

    # 遍历 CSV 文件中的文件名，检查是否存在于当前文件夹下
    missing_files = []
    for file in csv_files:
        if not os.path.exists(os.path.join(folder_path, file)):
            missing_files.append(file)

    return missing_files


if __name__ == '__main__':
    csv_file = '/home7/yilizhati/projects/data/FileName.csv'  # CSV 文件路径

    # source_folder = '/media/data1/jiachuang/data/medical/301segmentation_singletumor_t/train/DWI'  # 源文件夹路径
    # target_folder = '/media/data1/jiachuang/data/medical/301segmentation_singletumor_t/train/DWI/unused'  # 目标文件夹路径

    source_folder = '/home7/yilizhati/data/301kidney_tumor/singletumor_train/label/DWI'  # 源文件夹路径
    target_folder = '/home7/yilizhati/data/301kidney_tumor/singletumor_train/label/DWI/unused'  # 目标文件夹路径


    move_files_not_in_csv(source_folder, csv_file, target_folder)

    folder_path = '/home7/yilizhati/data/301kidney_tumor/singletumor_train/label/DWI'  #文件夹路径
    # folder_path = '/media/data1/jiachuang/data/medical/301segmentation_singletumor_t/label/DWI'
    niigz_count = count_niigz_files(folder_path)
    print("Number of NIfTI-GZ files:", niigz_count)

    missing_niigz = find_missing_niigz(csv_file, folder_path)
    print("Missing NIfTI-GZ files in", folder_path, ":", missing_niigz)

