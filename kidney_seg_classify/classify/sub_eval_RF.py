import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import joblib
import seaborn as sns
import shap
from statsmodels.stats.proportion import proportion_effectsize

# 创建输出文件夹
os.makedirs('final_performance', exist_ok=True)

# 读取测试数据
print("Loading test data...")
test_file_path = "../sub_data/features/test_features.csv"
data_test = pd.read_csv(test_file_path)
data_test = data_test.fillna(0)

# 分离特征和标签
X_test = data_test.iloc[:, 2:]
y_test = data_test.iloc[:, 1]

# 加载模型和预处理器
model_folder = os.path.join('trained_models', "random_forest_")
rf_model_final = joblib.load(os.path.join(model_folder, 'final_random_forest_model.joblib'))
var_selector = joblib.load(os.path.join(model_folder, 'var_selector.joblib'))
mi_selector = joblib.load(os.path.join(model_folder, 'mi_selector.joblib'))
ss = joblib.load(os.path.join(model_folder, 'standard_scaler.joblib'))
final_features = joblib.load(os.path.join(model_folder, 'final_features.joblib'))

# 加载最佳参数
best_params_final = joblib.load(os.path.join('final_best_params', 'best_params.joblib'))

# 预处理测试数据
X_test_scaled = ss.transform(X_test)
X_test_var = var_selector.transform(X_test_scaled)
X_test_mi = mi_selector.transform(X_test_var)
X_test_final = pd.DataFrame(X_test_mi, columns=X_test.columns[var_selector.get_support()][mi_selector.get_support()])[final_features]

# 在测试集上进行预测
print("Making predictions on test set...")
rf_prob_final = rf_model_final.predict_proba(X_test_final)[:, 1]


# 设定最佳阈值
optimal_threshold = 0.6328

# 使用最佳阈值进行分类
rf_pred_final = (rf_prob_final >= optimal_threshold).astype(int)

# 创建一个包含预测结果的 DataFrame
predictions_df = pd.DataFrame({
    'True_Label': y_test,
    'Predicted_Label': rf_pred_final,
    'Predicted_Probability': rf_prob_final
})

# 如果测试集中有 ID 列，你也可以将其添加到预测结果中
if 'ID' in data_test.columns:
    predictions_df['ID'] = data_test['ID'].values

os.makedirs('sub_final_performance', exist_ok=True)
# 保存预测结果到 CSV 文件
predictions_path = os.path.join('sub_final_performance', 'test_predictions.csv')
predictions_df.to_csv(predictions_path, index=False)
print(f"Test set predictions saved to {predictions_path}")

# 评估模型
rf_confusion_matrix_final = confusion_matrix(y_test, rf_pred_final)
rf_accuracy_final = (rf_confusion_matrix_final[0, 0] + rf_confusion_matrix_final[1, 1]) / len(y_test)
rf_sensitivity_final = rf_confusion_matrix_final[1, 1] / (rf_confusion_matrix_final[1, 1] + rf_confusion_matrix_final[1, 0])
rf_specificity_final = rf_confusion_matrix_final[0, 0] / (rf_confusion_matrix_final[0, 1] + rf_confusion_matrix_final[0, 0])
rf_auc_final = roc_auc_score(y_test, rf_prob_final)
print("Predictions completed.")

# 计算准确率的95%置信区间
n = len(y_test)
p = rf_accuracy_final
se = np.sqrt((p*(1-p))/n)
ci_95 = 1.96 * se
ci_lower = p - ci_95
ci_upper = p + ci_95
print(f"Accuracy: {rf_accuracy_final:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")

# 打印结果
print("Final Random Forest Performance:")

print(f"Accuracy: {rf_accuracy_final:.4f}")
print(f"Sensitivity: {rf_sensitivity_final:.4f}")
print(f"Specificity: {rf_specificity_final:.4f}")
print(f"AUC: {rf_auc_final:.4f}")

# 打印混淆矩阵
print("\nConfusion Matrix:")
print(f"True Negative (TN): {rf_confusion_matrix_final[0, 0]}")
print(f"False Positive (FP): {rf_confusion_matrix_final[0, 1]}")
print(f"False Negative (FN): {rf_confusion_matrix_final[1, 0]}")
print(f"True Positive (TP): {rf_confusion_matrix_final[1, 1]}")

# 特征重要性
feature_importance_final = pd.DataFrame({'feature': final_features, 'importance': rf_model_final.feature_importances_})
feature_importance_final = feature_importance_final.sort_values('importance', ascending=False)
print("\nTop 10 Important Features:")
print(feature_importance_final.head(10))

# 保存模型性能指标、最佳参数和特征重要性到一个文本文件
performance_summary_path = os.path.join('sub_final_performance', 'model_summary.txt')
with open(performance_summary_path, 'w') as f:
    f.write("Random Forest Model Summary\n")
    f.write("===========================\n\n")
    f.write("1. Model Performance:\n")
    f.write(f"   Accuracy: {rf_accuracy_final:.4f}\n")
    f.write(f"   Sensitivity: {rf_sensitivity_final:.4f}\n")
    f.write(f"   Specificity: {rf_specificity_final:.4f}\n")
    f.write(f"   AUC: {rf_auc_final:.4f}\n\n")
    
    f.write("2. Best Model Parameters:\n")
    for param, value in best_params_final.items():
        f.write(f"   {param}: {value}\n")
    f.write("\n")
    f.write(f"   Optimal Threshold: {optimal_threshold}\n\n")
    
    f.write("3. Feature Importance (Top 20):\n")
    for _, row in feature_importance_final.head(20).iterrows():
        f.write(f"   {row['feature']}: {row['importance']:.4f}\n")

print(f"Model summary saved to {performance_summary_path}")

# 将ROC曲线的每个值保存到CSV文件
fpr_rf_final, tpr_rf_final, thresholds_rf_final = roc_curve(y_test, rf_prob_final)
roc_values_df = pd.DataFrame({
    'False Positive Rate': fpr_rf_final,
    'True Positive Rate': tpr_rf_final,
    'Thresholds': thresholds_rf_final
})
roc_values_path = os.path.join('sub_final_performance', 'roc_value.csv')
roc_values_df.to_csv(roc_values_path, index=False)
print(f"ROC curve values saved to {roc_values_path}")

# ==================== 绘制ROC曲线 ====================
print("Creating ROC curve...")
plt.figure(figsize=(10, 8))
plt.plot(fpr_rf_final, tpr_rf_final, label=f'Random Forest (AUC = {rf_auc_final:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
# 保存ROC曲线图
roc_path = os.path.join('sub_final_performance', 'roc_curve.png')
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"ROC curve saved to {roc_path}")

# ==================== 绘制混淆矩阵 ====================
print("Creating confusion matrix...")
plt.figure(figsize=(10, 8))
sns.heatmap(rf_confusion_matrix_final, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
# 保存混淆矩阵图
cm_path = os.path.join('sub_final_performance', 'confusion_matrix.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Confusion matrix saved to {cm_path}")

# ==================== 特征重要性图 ====================
print("Creating feature importance plot...")
plt.figure(figsize=(20, 10))
top_13_features = feature_importance_final.head(13)
plt.barh(range(len(top_13_features)), top_13_features['importance'], align='center', height=0.6)
plt.yticks(range(len(top_13_features)), top_13_features['feature'], fontsize=15)
for i, v in enumerate(top_13_features['importance']):
    plt.text(v, i, f' {v:.4f}', va='center', fontsize=15)
plt.gca().invert_yaxis()
plt.title('Feature Importance', fontsize=18)
plt.xlabel('Importance', fontsize=14)
plt.xlim(0, max(top_13_features['importance']) * 1.1)
plt.tight_layout(pad=4.0)
plt.box(on=True)
# 保存特征重要性图
feature_importance_path = os.path.join('sub_final_performance', 'feature_importance.png')
plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Feature importance plot saved to {feature_importance_path}")

# ==================== SHAP分析 ====================
print("Calculating SHAP values...")
explainer = shap.TreeExplainer(rf_model_final)
shap_values = explainer.shap_values(X_test_final)

# 确定正确的SHAP值
if isinstance(shap_values, list):
    shap_values_to_plot = shap_values[1]  # 对于二分类，取正类的SHAP值
else:
    if shap_values.ndim == 3:
        shap_values_to_plot = shap_values[:, :, 1]
    else:
        shap_values_to_plot = shap_values

# 计算平均绝对SHAP值
shap_importance = np.abs(shap_values_to_plot).mean(0)
feature_importance_shap = pd.DataFrame({'feature': X_test_final.columns, 'importance': shap_importance})
feature_importance_shap = feature_importance_shap.sort_values('importance', ascending=False)

# ==================== SHAP摘要图 ====================
print("Creating SHAP summary plot...")
plt.figure(figsize=(50, 10))

# 只选择前13个最重要的特征
top_features = feature_importance_shap.head(13)['feature'].tolist()
X_sorted = X_test_final[top_features]

# 获取对应的SHAP值
indices = [list(X_test_final.columns).index(feat) for feat in top_features]
shap_values_sorted = shap_values_to_plot[:, indices]

plt.gcf().set_size_inches(50, 10)

# 创建SHAP摘要点图
shap.summary_plot(
    shap_values_sorted,
    X_sorted,
    plot_type="dot",
    show=False,
    max_display=13,  # 限制显示的特征数量
    alpha=0.8,      # 点的透明度
    cmap="RdBu_r",
    plot_size=(15, 8) 
)

# 设置标题和标签字体
plt.title('SHAP Feature Importance', fontsize=15)
plt.xlabel('SHAP value (impact on model output)', fontsize=12)
plt.yticks(fontsize=10)
plt.xlim(-0.4, 0.2)
plt.tight_layout(pad=4.0)

# 保存SHAP摘要图
shap_summary_path = os.path.join('sub_final_performance', 'shap_summary_plot.png')
plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"SHAP summary plot saved to {shap_summary_path}")

# 将SHAP特征重要性添加到性能摘要中
with open(performance_summary_path, 'a') as f:
    f.write("\n4. SHAP Feature Importance:\n")
    f.write("   Importance Values:\n")
    for _, row in feature_importance_shap.head(20).iterrows():
        f.write(f"   {row['feature']}: {row['importance']:.4f}\n")

print("All results, plots, and model evaluations have been saved successfully.")