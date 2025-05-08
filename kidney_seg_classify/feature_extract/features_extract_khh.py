import SimpleITK
import cv2
import scipy
from radiomics import featureextractor
# import nrrd
import numpy as np
import pandas
import radiomics
import os

file_path = r""
mask_path = r""
params_path = r"Params_myself1.yaml"
output_path = r""

extractor = featureextractor.RadiomicsFeatureExtractor(params_path)
table = pandas.DataFrame()
img = SimpleITK.ReadImage(file_path)
mask = SimpleITK.ReadImage(mask_path)
featureVector = pandas.Series(extractor.execute(img, mask))
featureVector.name = PicName[0:position]
table = table.join(featureVector, how='outer')
table1 = table.T
drop_dp = table1.filter(regex=('diagnostics.*'))

table1 = table1.drop(drop_dp.columns, axis=1)#删除列中的drop_dp.columns栏
table1.to_csv(output_path)