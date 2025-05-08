import sys
sys.path.append('..')

import os
import pydicom
import numpy as np
import SimpleITK as sitk
import nibabel as nib


def dicom_to_nifti(dicom_folder, nifti_file):
    # 1.构建dicom序列文件阅读器，并执行（即将dicom序列文件“打包整合”）
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_names)
    image2 = reader.Execute()
    # 2.将整合后的数据转为array，并获取dicom文件基本信息
    image_array = sitk.GetArrayFromImage(image2)  # z, y, x
    origin = image2.GetOrigin()  # x, y, z
    spacing = image2.GetSpacing()  # x, y, z
    direction = image2.GetDirection()  # x, y, z
    print(origin, spacing, direction)
    # 3.将array转为img，并保存为.nii.gz
    image3 = sitk.GetImageFromArray(image_array)
    image3.SetSpacing((1.0, 1.0, 1.0))
    image3.SetDirection(direction)
    image3.SetOrigin(origin)
    sitk.WriteImage(image3, nifti_file)

    print("Image conversion completed successfully.")


def resample_label(input_label_file, output_label_file, new_spacing=(1.0, 1.0, 1.0)):
    # 加载标签文件
    sitk_label = sitk.ReadImage(input_label_file)
    
    # 获取原始标签文件的空间信息
    original_spacing = sitk_label.GetSpacing()
    original_spacing_list = [float(s) for s in original_spacing]

    # 设置新的体素间距
    sitk_label.SetSpacing(new_spacing)

    # 保存重新采样后的标签数据为NIfTI文件
    sitk.WriteImage(sitk_label, output_label_file)

    print("Label resampling completed successfully.")


def transpose_nii(nii_file, output_nii_file, new_order):
    """
    将NIfTI图像的维度重新排列为指定顺序，并保存为新的NIfTI文件。

    参数:
    - nii_file (str): 输入NIfTI图像文件路径。
    - output_nii_file (str): 输出NIfTI图像文件路径。
    - new_order (tuple): 新的维度顺序，例如 (2, 0, 1)。
    """
    # 加载NIfTI图像
    nii_img = nib.load(nii_file)

    # 获取图像数据和头信息
    data = nii_img.get_fdata()
    header = nii_img.header

    # 调整维度顺序
    data_transposed = np.transpose(data, new_order)

    # 创建调整后的NIfTI图像对象
    nii_img_transposed = nib.Nifti1Image(data_transposed, None, header)

    # 保存调整后的NIfTI图像
    nib.save(nii_img_transposed, output_nii_file)

    print("Image dimension transpose completed successfully.")


if __name__ == '__main__':
    # 输入DICOM文件夹路径和输出NIfTI文件路径

    dicom_folder1 = '../data/CRMs/Y7211977.nii.gz/ADC/'
    os.makedirs('../data/target/ADC/image', exist_ok=True)
    nifti_file1 = '../data/target/ADC/image/Y7211977.nii.gz'

    dicom_folder2 = '../data/CRMs/Y7211977.nii.gz/CP/'
    os.makedirs('../data/target/CP/image', exist_ok=True)
    nifti_file2 = '../data/target/CP/image/Y7211977.nii.gz'

    dicom_folder3 = '../data/CRMs/Y7211977.nii.gz/DWI/'
    os.makedirs('../data/target/DWI/image', exist_ok=True)
    nifti_file3 = '../data/target/DWI/image/Y7211977.nii.gz'

    dicom_folder4 = '../data/CRMs/Y7211977.nii.gz/EP/'
    os.makedirs('../data/target/EP/image', exist_ok=True)
    nifti_file4 = '../data/target/EP/image/Y7211977.nii.gz'

    dicom_folder5 = '../data/CRMs/Y7211977.nii.gz/NP/'
    os.makedirs('../data/target/NP/image', exist_ok=True)
    nifti_file5 = '../data/target/NP/image/Y7211977.nii.gz'

    dicom_folder6 = '../data/CRMs/Y7211977.nii.gz/T1WI/'
    os.makedirs('../data/target/T1WI/image', exist_ok=True)
    nifti_file6 = '../data/target/T1WI/image/Y7211977.nii.gz'

    dicom_folder7 = '../data/CRMs/Y7211977.nii.gz/T2WI/'
    os.makedirs('../data/target/T2WI/image', exist_ok=True)
    nifti_file7 = '../data/target/T2WI/image/Y7211977.nii.gz'
    

    # 输入label文件夹路径和输出label文件路径
    input_label_file1 = '../data/CRMs/Y7211977.nii.gz/ADC/ADC.nii.gz'
    os.makedirs('../data/target/ADC/label', exist_ok=True)
    output_label_file1 = '../data/target/ADC/label/Y7211977.nii.gz'

    input_label_file2 = '../data/CRMs/Y7211977.nii.gz/CP/CP.nii.gz'
    os.makedirs('../data/target/CP/label', exist_ok=True)
    output_label_file2 = '../data/target/CP/label/Y7211977.nii.gz'

    input_label_file3 = '../data/CRMs/Y7211977.nii.gz/DWI/DWI.nii.gz'
    os.makedirs('../data/target/DWI/label', exist_ok=True)
    output_label_file3 = '../data/target/DWI/label/Y7211977.nii.gz'

    input_label_file4 = '../data/CRMs/Y7211977.nii.gz/EP/EP.nii.gz'
    os.makedirs('../data/target/EP/label', exist_ok=True)
    output_label_file4 = '../data/target/EP/label/Y7211977.nii.gz'

    input_label_file5 = '../data/CRMs/Y7211977.nii.gz/NP/NP.nii.gz'
    os.makedirs('../data/target/NP/label', exist_ok=True)
    output_label_file5 = '../data/target/NP/label/Y7211977.nii.gz'

    input_label_file6 = '../data/CRMs/Y7211977.nii.gz/T1WI/T1WI.nii.gz'
    os.makedirs('../data/target/T1WI/label', exist_ok=True)
    output_label_file6 = '../data/target/T1WI/label/Y7211977.nii.gz'

    input_label_file7 = '../data/CRMs/Y7211977.nii.gz/T2WI/T2WI.nii.gz'
    os.makedirs('../data/target/T2WI/label', exist_ok=True)
    output_label_file7 = '../data/target/T2WI/label/Y7211977.nii.gz'
    
    # #将dicom转为nii.gz格式
    dicom_to_nifti(dicom_folder1, nifti_file1)
    dicom_to_nifti(dicom_folder2, nifti_file2)
    dicom_to_nifti(dicom_folder3, nifti_file3)
    dicom_to_nifti(dicom_folder4, nifti_file4)
    dicom_to_nifti(dicom_folder5, nifti_file5)
    dicom_to_nifti(dicom_folder6, nifti_file6)
    dicom_to_nifti(dicom_folder7, nifti_file7)
    resample_label(input_label_file1,output_label_file1)
    resample_label(input_label_file2,output_label_file2)
    resample_label(input_label_file3,output_label_file3)
    resample_label(input_label_file4,output_label_file4)
    resample_label(input_label_file5,output_label_file5)
    resample_label(input_label_file6,output_label_file6)
    resample_label(input_label_file7,output_label_file7)

    # 标签体素转换 x y z---> z x y    (记得改回绝对路径，看学长原代码/home7/yilizhati/data/301kidney_tumor/singletumor_trian/label/ADC/Y7211977.nii.gz)
    nii_file1 = '../data/target/ADC/label/Y7211977.nii.gz'
    output_nii_file1 = '../data/singletumor_trian/label/ADC/Y7211977.nii.gz'   #存至项目目录下
    output_nii_file12 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/label/ADC/Y7211977.nii.gz'   #存至数据目录下
 
    nii_file2 = '../data/target/CP/label/Y7211977.nii.gz'                 
    output_nii_file2 = '../data/singletumor_trian/label/corticomedullary phase/Y7211977.nii.gz'
    output_nii_file22 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/label/corticomedullary phase/Y7211977.nii.gz'
  
    nii_file3 = '../data/target/DWI/label/Y7211977.nii.gz'
    output_nii_file3 = '../data/singletumor_trian/label/DWI/Y7211977.nii.gz'
    output_nii_file32 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/label/DWI/Y7211977.nii.gz'
 
    nii_file4 = '../data/target/EP/label/Y7211977.nii.gz'
    output_nii_file4 = '../data/singletumor_trian/label/excretory phase/Y7211977.nii.gz'
    output_nii_file42 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/label/excretory phase/Y7211977.nii.gz'
  
    nii_file5 = '../data/target/NP/label/Y7211977.nii.gz'
    output_nii_file5 = '../data/singletumor_trian/label/nephrographic phase/Y7211977.nii.gz'
    output_nii_file52 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/label/nephrographic phase/Y7211977.nii.gz'
  
    nii_file6 = '../data/target/T1WI/label/Y7211977.nii.gz'
    output_nii_file6 = '../data/singletumor_trian/label/T1WI/Y7211977.nii.gz'
    output_nii_file62 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/label/T1WI/Y7211977.nii.gz'
  
    nii_file7 = '../data/target/T2WI/label/Y7211977.nii.gz'
    output_nii_file7 = '../data/singletumor_trian/label/T2WI/Y7211977.nii.gz'
    output_nii_file72 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/label/T2WI/Y7211977.nii.gz'
    #加两行转换数据
    new_order = (2, 0, 1)  # 新的维度顺序
    transpose_nii(nii_file1, output_nii_file1, new_order)
    transpose_nii(nii_file1, output_nii_file12, new_order)    
    transpose_nii(nii_file2, output_nii_file2, new_order)
    transpose_nii(nii_file2, output_nii_file22, new_order)
    transpose_nii(nii_file3, output_nii_file3, new_order)
    transpose_nii(nii_file3, output_nii_file32, new_order)
    transpose_nii(nii_file4, output_nii_file4, new_order)
    transpose_nii(nii_file4, output_nii_file42, new_order)
    transpose_nii(nii_file5, output_nii_file5, new_order)
    transpose_nii(nii_file5, output_nii_file52, new_order)
    transpose_nii(nii_file6, output_nii_file6, new_order)
    transpose_nii(nii_file6, output_nii_file62, new_order)
    transpose_nii(nii_file7, output_nii_file7, new_order)
    transpose_nii(nii_file7, output_nii_file72, new_order)

    # 数据体素转换 x y z---> z x y    (记得改回绝对路径，看学长原代码/home7/yilizhati/data/301kidney_tumor/singletumor_trian/label/ADC/Y7211977.nii.gz)
    nii_file1 = '../data/target/ADC/image/Y7211977.nii.gz'
    output_nii_file1 = '../data/singletumor_trian/image/ADC/Y7211977.nii.gz'   #存至项目目录下
    output_nii_file12 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/image/ADC/Y7211977.nii.gz'   #存至数据目录下
 
    nii_file2 = '../data/target/CP/image/Y7211977.nii.gz'                      
    output_nii_file2 = '../data/singletumor_trian/image/corticomedullary phase/Y7211977.nii.gz'
    output_nii_file22 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/image/corticomedullary phase/Y7211977.nii.gz'
  
    nii_file3 = '../data/target/DWI/image/Y7211977.nii.gz'
    output_nii_file3 = '../data/singletumor_trian/image/DWI/Y7211977.nii.gz'
    output_nii_file32 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/image/DWI/Y7211977.nii.gz'
 
    nii_file4 = '../data/target/EP/image/Y7211977.nii.gz'
    output_nii_file4 = '../data/singletumor_trian/image/excretory phase/Y7211977.nii.gz'
    output_nii_file42 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/image/excretory phase/Y7211977.nii.gz'
  
    nii_file5 = '../data/target/NP/image/Y7211977.nii.gz'
    output_nii_file5 = '../data/singletumor_trian/image/nephrographic phase/Y7211977.nii.gz'
    output_nii_file52 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/image/nephrographic phase/Y7211977.nii.gz'
  
    nii_file6 = '../data/target/T1WI/image/Y7211977.nii.gz'
    output_nii_file6 = '../data/singletumor_trian/image/T1WI/Y7211977.nii.gz'
    output_nii_file62 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/image/T1WI/Y7211977.nii.gz'
  
    nii_file7 = '../data/target/T2WI/image/Y7211977.nii.gz'
    output_nii_file7 = '../data/singletumor_trian/image/T2WI/Y7211977.nii.gz'
    output_nii_file72 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/image/T2WI/Y7211977.nii.gz'
    #加两行转换数据
    new_order = (2, 0, 1)  # 新的维度顺序
    transpose_nii(nii_file1, output_nii_file1, new_order)
    transpose_nii(nii_file1, output_nii_file12, new_order)    
    transpose_nii(nii_file2, output_nii_file2, new_order)
    transpose_nii(nii_file2, output_nii_file22, new_order)
    transpose_nii(nii_file3, output_nii_file3, new_order)
    transpose_nii(nii_file3, output_nii_file32, new_order)
    transpose_nii(nii_file4, output_nii_file4, new_order)
    transpose_nii(nii_file4, output_nii_file42, new_order)
    transpose_nii(nii_file5, output_nii_file5, new_order)
    transpose_nii(nii_file5, output_nii_file52, new_order)
    transpose_nii(nii_file6, output_nii_file6, new_order)
    transpose_nii(nii_file6, output_nii_file62, new_order)
    transpose_nii(nii_file7, output_nii_file7, new_order)
    transpose_nii(nii_file7, output_nii_file72, new_order)


    # 打印标签体素间距
    label_file1 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/label/ADC/Y7211977.nii.gz'
    label_image = nib.load(label_file1)
    header = label_image.header
    size = label_image.shape
    print("ADC LABEL Voxel Spacing:", header.get_zooms(), "Size:", size)

    label_file2 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/label/corticomedullary phase/Y7211977.nii.gz'
    label_image = nib.load(label_file2)
    header = label_image.header
    size = label_image.shape
    print("CP LABEL Voxel Spacing:", header.get_zooms(), "Size:", size)

    label_file3 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/label/DWI/Y7211977.nii.gz'
    label_image = nib.load(label_file3)
    header = label_image.header
    size = label_image.shape
    print("DWI LABEL Voxel Spacing:", header.get_zooms(), "Size:", size)

    label_file4 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/label/excretory phase/Y7211977.nii.gz'
    label_image = nib.load(label_file4)
    header = label_image.header
    size = label_image.shape
    print("EP LABEL Voxel Spacing:", header.get_zooms(), "Size:", size)

    label_file5 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/label/nephrographic phase/Y7211977.nii.gz'
    label_image = nib.load(label_file5)
    header = label_image.header
    size = label_image.shape
    print("NP LABEL Voxel Spacing:", header.get_zooms(), "Size:", size)

    label_file6 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/label/T1WI/Y7211977.nii.gz'
    label_image = nib.load(label_file6)
    header = label_image.header
    size = label_image.shape
    print("T1WI Label Voxel Spacing:", header.get_zooms(), "Size:", size)

    label_file7 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/label/T2WI/Y7211977.nii.gz'
    label_image = nib.load(label_file7)
    header = label_image.header
    size = label_image.shape
    print("T2WI Label Voxel Spacing:", header.get_zooms(), "Size:", size)

    # 打印数据体素间距
    label_file1 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/image/ADC/Y7211977.nii.gz'
    label_image = nib.load(label_file1)
    header = label_image.header
    size = label_image.shape
    print("ADC Voxel Spacing:", header.get_zooms(), "Size:", size)

    label_file2 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/image/corticomedullary phase/Y7211977.nii.gz'
    label_image = nib.load(label_file2)
    header = label_image.header
    size = label_image.shape
    print("CP Voxel Spacing:", header.get_zooms(), "Size:", size)

    label_file3 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/image/DWI/Y7211977.nii.gz'
    label_image = nib.load(label_file3)
    header = label_image.header
    size = label_image.shape
    print("DWI Voxel Spacing:", header.get_zooms(), "Size:", size)

    label_file4 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/image/excretory phase/Y7211977.nii.gz'
    label_image = nib.load(label_file4)
    header = label_image.header
    size = label_image.shape
    print("EP Voxel Spacing:", header.get_zooms(), "Size:", size)

    label_file5 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/image/nephrographic phase/Y7211977.nii.gz'
    label_image = nib.load(label_file5)
    header = label_image.header
    size = label_image.shape
    print("NP Voxel Spacing:", header.get_zooms(), "Size:", size)

    label_file6 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/image/T1WI/Y7211977.nii.gz'
    label_image = nib.load(label_file6)
    header = label_image.header
    size = label_image.shape
    print("T1WI Voxel Spacing:", header.get_zooms(), "Size:", size)

    label_file7 = '/home7/yilizhati/data/301kidney_tumor/singletumor_trian/image/T2WI/Y7211977.nii.gz'
    label_image = nib.load(label_file7)
    header = label_image.header
    size = label_image.shape
    print("T2WI Voxel Spacing:", header.get_zooms(), "Size:", size)

