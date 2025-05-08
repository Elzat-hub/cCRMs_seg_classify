import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest
import warnings
import joblib
import random

warnings.filterwarnings('ignore')

# 固定随机种子
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# 设置GPU
GPU_id = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Using CUDA device {GPU_id}: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

# 定义随机森林函数
def rf(X_train, y_train):
    print("Starting Random Forest model training...")
    model_forest = RandomForestClassifier(random_state=seed)

    param_grid = {
        "n_estimators": [2,5,10,15,20],
        "criterion": ["gini", "entropy"],
        "max_features": ["sqrt", "log2", None, 1, 3, 4, 5, 6,7,8,10],
        "max_depth": [7, 9, 10,11, 12,13],
        "min_samples_split": [10,12,14,16,15],
        'class_weight' : [{0:0.3, 1:0.75}]
    }

    cv = KFold(n_splits=10, shuffle=True, random_state=seed)
    grid_search = GridSearchCV(model_forest, param_grid, n_jobs=-1, scoring='accuracy', cv=cv)
    grid_search.fit(X_train, y_train)

    model_forest = RandomForestClassifier(
        n_estimators=grid_search.best_params_["n_estimators"],
        criterion=grid_search.best_params_["criterion"],
        max_features=grid_search.best_params_["max_features"],
        max_depth=grid_search.best_params_["max_depth"],
        min_samples_split=grid_search.best_params_["min_samples_split"],
        class_weight=grid_search.best_params_["class_weight"],
        random_state=seed
    ).fit(X_train, y_train)

    print("Random Forest model training completed.")
    return model_forest, grid_search.best_params_

# 读取数据
print("Loading data...")
train_file_path = "../data/features/train_features.csv"
data_train = pd.read_csv(train_file_path)
data_train = data_train.fillna(0)

# 分离特征和标签
X_train = data_train.iloc[:, 2:]
y_train = data_train.iloc[:, 1]

# 标准化
print("Standardizing data...")
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# 特征选择
print("Performing feature selection...")
var_selector = VarianceThreshold(threshold=0.01)
X_train_var = var_selector.fit_transform(X_train_scaled)

# 互信息选择
mi_selector = SelectKBest(mutual_info_classif, k=80)
X_train_mi = mi_selector.fit_transform(X_train_var, y_train)

# LASSO特征选择
lasso = LassoCV(alphas=np.logspace(-4, 0, 50), cv=5, max_iter=100000, random_state=seed)
lasso.fit(X_train_mi, y_train)

# 选择LASSO系数绝对值大于某个阈值的特征
lasso_coef = pd.Series(lasso.coef_, index=X_train.columns[var_selector.get_support()][mi_selector.get_support()])
final_features = lasso_coef[abs(lasso_coef) > 0.01].index
print(f"Number of features selected: {len(final_features)}")

# 更新训练集
X_train_final = pd.DataFrame(X_train_mi, columns=X_train.columns[var_selector.get_support()][mi_selector.get_support()])[final_features]

# 训练最终的随机森林模型
rf_model_final, best_params_final = rf(X_train_final, y_train)

# 创建必要的目录
directories = ['trained_models', 'final_performance', 'final_best_params']
for directory in directories:
    os.makedirs(directory, exist_ok=True)

model_name = "random_forest"
model_folder = os.path.join('trained_models', f"{model_name}_")
os.makedirs(model_folder, exist_ok=True)

# 保存训练好的最终随机森林模型
print("Saving the trained Random Forest model...")
model_path = os.path.join(model_folder, f'final_random_forest_model.joblib')
joblib.dump(rf_model_final, model_path)
print(f"Final Random Forest model saved to {model_path}")

# 保存特征选择器和标准化器
joblib.dump(var_selector, os.path.join(model_folder, 'var_selector.joblib'))
joblib.dump(mi_selector, os.path.join(model_folder, 'mi_selector.joblib'))
joblib.dump(ss, os.path.join(model_folder, 'standard_scaler.joblib'))
joblib.dump(final_features, os.path.join(model_folder, 'final_features.joblib'))

# 保存最佳参数
best_params_path = os.path.join('final_best_params', 'best_params.joblib')
joblib.dump(best_params_final, best_params_path)
print(f"Best parameters saved to {best_params_path}")

print("Model training, feature selection, and saving completed.")