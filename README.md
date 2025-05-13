# cCRMs_seg_classify
The cCRMs_seg_classify project is used for a Machine Learning Pipeline for Automated Segmentation and Classification of Complicated Cystic Renal Masses in MRI.

# Installing Dependencies
```bash
pip install -r requirements.txt
```

# Data Preprocessing
The input images required for model creation, along with data augmentation operations, are conducted offline.
```bash
python [certain modality]_process.py
```

# Segmentation Training
Using the default values for hyper-parameters, the following command can be used to initiate training using PyTorch:
```bash
python train_[certain modality].py
--batch_size=2
--nEpochs=500
--classes=2
--inChannels=1
--dropout_rate=0.5
--lr=1e-3
--opt='adamw'
--model='VNET' or 'UNET3D'
--cuda
```

# Segmentation Validation
You can use the following command to initiate model inference using PyTorch:
```bash
python eval_[certain modality].py
--batchSz=1
--classes=2
--inChannels=1
--dropout_rate=0.5
--model='VNET' or 'UNET3D'
--resume='model_saved_path/model.pth'
--cuda
```
# Feature Extract
```bash
python feature_extract.py
```
# Training Classifier
Using the default values for hyperparameters, you can use the following command to train a random forest.
```bash
RandomForestClassifier 超参数搜索空间：
{
  "n_estimators": [2,5,10,15,20],
  "criterion": ["gini", "entropy"],
  "max_features": ["sqrt", "log2", None, 1, 3-10],
  "max_depth": [7-13],
  "min_samples_split": [10-16],
  "class_weight": [{0:0.3, 1:0.75}]
}
```
# Evaluate Classifier
Evaluate the random forest model using the following command.
```bash
python eval_RF.py
```

