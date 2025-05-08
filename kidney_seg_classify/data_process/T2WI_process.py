import sys
sys.path.append('..')

from monai import transforms
import numpy as np
import os
import glob
import nibabel as nib

# Single tumor T2WI MRI:190 ~ 1600 Set:150 ~ 1800 Size:48x256x256
train_path = "/home7/yilizhati/data/301kidney_tumor/singletumor_train/image/T2WI"
train_label_path = "/home7/yilizhati/data/301kidney_tumor/singletumor_train/label/T2WI"
val_path = "/home7/yilizhati/data/301kidney_tumor/singletumor_val/image/T2WI"
val_label_path = "/home7/yilizhati/data/301kidney_tumor/singletumor_val/label/T2WI"

def normalization_threshold(data, threshold=1.0): 
    data = data.astype(float) # 确保数据是浮点数类型 
    data[data >= threshold] = 1
    data[data < threshold] = 0
    return data

if __name__ == '__main__':

    list_train = sorted(glob.glob(os.path.join(train_path, '*.nii.gz')))
    label_train = sorted(glob.glob(os.path.join(train_label_path, '*.nii.gz')))
    list_val = sorted(glob.glob(os.path.join(val_path, '*.nii.gz')))
    label_val = sorted(glob.glob(os.path.join(val_label_path, '*.nii.gz')))

    train_data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(list_train, label_train)
    ]

    val_data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(list_val, label_val)
    ]

    original_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.CenterSpatialCropd(
                keys=["image", "label"],
                roi_size=(48, 256, 256),
            ),
            transforms.ScaleIntensityRanged(keys=["image"],
                                            a_min=150.0,
                                            a_max=1800.0,
                                            b_min=0.0,
                                            b_max=1.0,
                                            clip=True),
            transforms.SpatialPadD(keys=["image", "label"],
                                   spatial_size=(48, 256, 256),
                                   method='symmetric',
                                   mode='constant'),
        ]
    )

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"],
                                    axcodes="RAS"),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.CenterSpatialCropd(
                keys=["image", "label"],
                roi_size=(48, 256, 256),
            ),
            transforms.RandFlipd(keys=["image", "label"],
                                 prob=0.8,
                                 spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"],
                                 prob=0.7,
                                 spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"],
                                 prob=0.6,
                                 spatial_axis=2),
            transforms.RandRotate90d(
                keys=["image", "label"],
                prob=0.5,
                max_k=3,
                spatial_axes=(1, 2)
            ),
            transforms.RandScaleIntensityd(keys="image",
                                           factors=0.1,
                                           prob=0.5),
            transforms.RandShiftIntensityd(keys="image",
                                           offsets=0.1,
                                           prob=0.5),

            transforms.ScaleIntensityRanged(keys=["image"],
                                            a_min=150.0,
                                            a_max=1800.0,
                                            b_min=0.0,
                                            b_max=1.0,
                                            clip=True),
            transforms.SpatialPadD(keys=["image", "label"],
                                   spatial_size=(48, 256, 256),
                                   method='symmetric',
                                   mode='constant'),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.CropForegroundd(keys=["image", "label"],
                                       source_key="image"),
            transforms.CenterSpatialCropd(
                keys=["image", "label"],
                roi_size=(48, 256, 256),
            ),

            transforms.ScaleIntensityRanged(keys=["image"],
                                            a_min=150.0,
                                            a_max=1800.0,
                                            b_min=0.0,
                                            b_max=1.0,
                                            clip=True),
            transforms.SpatialPadD(keys=["image", "label"],
                                   spatial_size=(48, 256, 256),
                                   method='symmetric',
                                   mode='constant'),
        ]
    )
    print("====Start Augmentation====")

    # aug_train = train_transform(train_data_dicts)
    # ori_train = original_transform(train_data_dicts)
    aug_val = val_transform(val_data_dicts)
    loader = transforms.LoadImaged(keys=("image", "label"))

    # # training dataset augmentation Single tumor T2WI
    # for i in range(len(list_train)):
    #     img_save = aug_train[i]['image'].squeeze(0).detach().cpu().numpy()
    #     lab_save = aug_train[i]['label'].squeeze(0).detach().cpu().numpy()
    #     lab_save = normalization_threshold(lab_save,threshold=1.0)
    #     print(f"t_a image shape: {img_save.shape}", "type:", type(img_save), "max_min", aug_train[i]['image'].max(),
    #           aug_train[i]['image'].min())
    #     print(f"t_a lab shape: {lab_save.shape}", type(lab_save), "max_min", aug_train[i]['label'].max(),
    #           aug_train[i]['label'].min())
    #     new_img = nib.Nifti1Image(img_save, np.eye(4))
    #     new_lab = nib.Nifti1Image(lab_save, np.eye(4))

    #     os.makedirs('../data/T2WI/train/image', exist_ok=True)
    #     os.makedirs('../data/T2WI/train/label', exist_ok=True)
    #     nib.save(new_img, '../data/T2WI/train/image/aug_{}.nii.gz'.format(list_train[i].split("/")[-1].split(".")[0]))
    #     nib.save(new_lab, '../data/T2WI/train/label/aug_{}.nii.gz'.format(list_train[i].split("/")[-1].split(".")[0]))
    # print("Training dataset augmentation transformed FINISH.")

    # # training dataset original crop Single tumor T2WI
    # for i in range(len(list_train)):
    #     img_save = ori_train[i]['image'].squeeze(0).detach().cpu().numpy()
    #     lab_save = ori_train[i]['label'].squeeze(0).detach().cpu().numpy()
    #     lab_save = normalization_threshold(lab_save,threshold=1.0)
    #     print(f"t_o image shape: {img_save.shape}", "type:", type(img_save), "max_min", ori_train[i]['image'].max(),
    #           ori_train[i]['image'].min())
    #     print(f"t_o lab shape: {lab_save.shape}", type(lab_save), "max_min", ori_train[i]['label'].max(),
    #           ori_train[i]['label'].min())
    #     new_img = nib.Nifti1Image(img_save, np.eye(4))
    #     new_lab = nib.Nifti1Image(lab_save, np.eye(4))
    #     os.makedirs('../data/T2WI/train/image', exist_ok=True)
    #     os.makedirs('../data/T2WI/train/label', exist_ok=True)
    #     nib.save(new_img, '../data/T2WI/train/image/ori_{}.nii.gz'.format(list_train[i].split("/")[-1].split(".")[0]))
    #     nib.save(new_lab, '../data/T2WI/train/label/ori_{}.nii.gz'.format(list_train[i].split("/")[-1].split(".")[0]))
    # print("Training dataset original cropped FINISH.")

    # valing dataset ori Single tumor T2WI     这是有用的，  可以跟上面的代码一起用，val的数据进行裁剪成统一  256*256的，   不同序列的大小看Vnet
    for i in range(len(list_val)):
        img_save = aug_val[i]['image'].squeeze(0).detach().cpu().numpy()
        lab_save = aug_val[i]['label'].squeeze(0).detach().cpu().numpy()
        lab_save = normalization_threshold(lab_save,threshold=1.0)
        print(f"v image shape: {img_save.shape}", "type:", type(img_save), "max_min", aug_val[i]['image'].max(),
              aug_val[i]['image'].min())
        print(f"v lab shape: {lab_save.shape}", type(lab_save), "max_min", aug_val[i]['label'].max(),
              aug_val[i]['label'].min())
        new_img = nib.Nifti1Image(img_save, np.eye(4))
        new_lab = nib.Nifti1Image(lab_save, np.eye(4))
        os.makedirs('../data/T2WI/val/image', exist_ok=True)
        os.makedirs('../data/T2WI/val/label', exist_ok=True)
        nib.save(new_img,
         '../data/T2WI/val/image/t_{}.nii.gz'.format(list_val[i].split("/")[-1].split(".")[0]))
        nib.save(new_lab,
         '../data/T2WI/val/label/t_{}.nii.gz'.format(list_val[i].split("/")[-1].split(".")[0]))
    print("validating dataset transformed FINISH.")
