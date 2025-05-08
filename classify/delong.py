import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.utils import resample

# Helper functions for DeLong test
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)  
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float) 
    T2[J] = T + 1
    return T2

def compute_auc(x, y):
    x = np.array(x)
    y = np.array(y)
    n1 = len(x)
    n2 = len(y)
    r = compute_midrank(np.concatenate((x, y)))
    aucs = (np.sum(r[:n1]) - n1 * (n1 + 1) / 2) / (n1 * n2)
    return aucs

def compute_var_auc(x, y):
    x = np.array(x)
    y = np.array(y)
    n1 = len(x)
    n2 = len(y)
    r = compute_midrank(np.concatenate((x, y)))
    s = r[:n1]
    t = r[n1:]
    var_auc = (np.sum(s**2) - n1*(n1+1)*(2*n1+1)/6) / (n1*n2)**2
    var_auc += (np.sum(t**2) - n2*(n2+1)*(2*n2+1)/6) / (n1*n2)**2
    var_auc /= (n1-1)*(n2-1)
    return var_auc

def compute_cov_auc(x1, x2, y1, y2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)
    
    n1 = len(x1)
    n2 = len(x2)
    
    r_x = compute_midrank(np.concatenate((x1, x2)))
    r_y = compute_midrank(np.concatenate((y1, y2)))
    
    s_x = r_x[:n1]
    s_y = r_y[:n1]
    
    cov_auc = (np.sum(s_x * s_y) - n1 * (n1 + 1) * (2 * n1 + 1) / 6) / (n1 * n2)
    cov_auc /= (n1 - 1) * (n2 - 1)
    return cov_auc

def delong_test(y_true, y1_pred, y2_pred):
    y_true = np.array(y_true)
    y1_pred = np.array(y1_pred)
    y2_pred = np.array(y2_pred)
    
    # Ensure y_true contains only 0 and 1
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true should only contain 0 and 1")
    
    x1 = y1_pred[y_true == 1]
    x2 = y1_pred[y_true == 0]
    y1 = y2_pred[y_true == 1]
    y2 = y2_pred[y_true == 0]
    
    auc1 = compute_auc(x2, x1)
    auc2 = compute_auc(y2, y1)
    
    n1 = len(x1)
    n2 = len(x2)
    
    var_auc1 = compute_var_auc(x2, x1)
    var_auc2 = compute_var_auc(y2, y1)
    print(f"Var AUC1: {var_auc1}")
    print(f"Var AUC2: {var_auc2}")

    cov_auc = compute_cov_auc(x1, x2, y1, y2)
    print(f"Cov AUC: {cov_auc}")

    delta_auc = auc1 - auc2
    var_delta_auc = var_auc1 + var_auc2 - 2*cov_auc
    
    z = delta_auc / np.sqrt(var_delta_auc)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return z, p, delta_auc

# Read data
# data = pd.read_excel("expert_label.xlsx", sheet_name="test set（60）")
data = pd.read_excel("sub_expert_label.xlsx", sheet_name="test set（60）")

# Print column names to ensure we're using the correct ones
print(data.columns)

# Extract pathological labels (gold standard), senior and junior expert scores
y_true = data['病理标签'].values
senior_expert_scores = data['高年资'].values
junior_expert_scores = data['低年资'].values

# Read random forest model's predicted probabilities
# rf_predictions = pd.read_csv('final_performance/test_predictions.csv')
rf_predictions = pd.read_csv('sub_final_performance/test_predictions.csv')
rf_prob_final = rf_predictions['Predicted_Probability'].values

# Ensure y_true only contains 0 and 1
y_true = np.array([1 if label == 1 else 0 for label in y_true])

# Print the count of each category
print("Number of positive cases:", np.sum(y_true == 1))
print("Number of negative cases:", np.sum(y_true == 0))

# Check if lengths are consistent
print("Length of y_true:", len(y_true))
print("Length of senior_expert_scores:", len(senior_expert_scores))
print("Length of junior_expert_scores:", len(junior_expert_scores))
print("Length of rf_prob_final:", len(rf_prob_final))

# Ensure lengths are consistent
assert len(y_true) == len(senior_expert_scores) == len(junior_expert_scores) == len(rf_prob_final), "Lengths are not consistent"

# Calculate ROC curves
fpr_senior, tpr_senior, _ = roc_curve(y_true, senior_expert_scores)
auc_senior = roc_auc_score(y_true, senior_expert_scores)

fpr_junior, tpr_junior, _ = roc_curve(y_true, junior_expert_scores)
auc_junior = roc_auc_score(y_true, junior_expert_scores)

fpr_rf, tpr_rf, _ = roc_curve(y_true, rf_prob_final)
auc_rf = roc_auc_score(y_true, rf_prob_final)

# Calculate 95% CI for AUC using bootstrapping
def bootstrap_auc_ci(y_true, y_score, n_bootstraps=2000, ci=0.95):
    bootstrapped_scores = []
    rng = np.random.RandomState(42)  # For reproducibility
    
    for i in range(n_bootstraps):
        # Bootstrap by sampling with replacement
        indices = rng.randint(0, len(y_score), len(y_score))
        if len(np.unique(y_true[indices])) < 2:
            # Skip if bootstrap sample has only one class
            continue
        score = roc_auc_score(y_true[indices], y_score[indices])
        bootstrapped_scores.append(score)
    
    # Sort the scores and get the confidence interval
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    # Compute confidence interval
    alpha = 1.0 - ci
    lower_bound = sorted_scores[int(alpha/2 * len(sorted_scores))]
    upper_bound = sorted_scores[int((1-alpha/2) * len(sorted_scores))]
    
    return lower_bound, upper_bound

# Calculate 95% CI for each AUC
auc_senior_lower, auc_senior_upper = bootstrap_auc_ci(y_true, senior_expert_scores)
auc_junior_lower, auc_junior_upper = bootstrap_auc_ci(y_true, junior_expert_scores)
auc_rf_lower, auc_rf_upper = bootstrap_auc_ci(y_true, rf_prob_final)

# Calculate metrics based on threshold (Bosniak 4 and 5 are considered malignant, 2 and 3 are benign)
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_auc_score

# For Senior Expert
senior_binary = (senior_expert_scores >= 4).astype(int)  # Scores 4 and 5 are considered malignant
senior_cm = confusion_matrix(y_true, senior_binary)
senior_tn, senior_fp, senior_fn, senior_tp = senior_cm.ravel()
senior_accuracy = accuracy_score(y_true, senior_binary)
senior_sensitivity = recall_score(y_true, senior_binary)  # Same as recall
senior_specificity = senior_tn / (senior_tn + senior_fp)

# Calculate 95% confidence intervals
n = len(y_true)
n_pos = np.sum(y_true == 1)  # Number of positive cases
n_neg = np.sum(y_true == 0)  # Number of negative cases

# For accuracy (Senior Expert)
p_senior_acc = senior_accuracy
se_senior_acc = np.sqrt((p_senior_acc * (1 - p_senior_acc)) / n)
ci_95_senior_acc = 1.96 * se_senior_acc
ci_lower_senior_acc = max(0, p_senior_acc - ci_95_senior_acc)
ci_upper_senior_acc = min(1, p_senior_acc + ci_95_senior_acc)

# For sensitivity (Senior Expert)
p_senior_sens = senior_sensitivity
se_senior_sens = np.sqrt((p_senior_sens * (1 - p_senior_sens)) / n_pos) if n_pos > 0 else 0
ci_95_senior_sens = 1.96 * se_senior_sens
ci_lower_senior_sens = max(0, p_senior_sens - ci_95_senior_sens)
ci_upper_senior_sens = min(1, p_senior_sens + ci_95_senior_sens)

# For specificity (Senior Expert)
p_senior_spec = senior_specificity
se_senior_spec = np.sqrt((p_senior_spec * (1 - p_senior_spec)) / n_neg) if n_neg > 0 else 0
ci_95_senior_spec = 1.96 * se_senior_spec
ci_lower_senior_spec = max(0, p_senior_spec - ci_95_senior_spec)
ci_upper_senior_spec = min(1, p_senior_spec + ci_95_senior_spec)

# For Junior Expert
junior_binary = (junior_expert_scores >= 4).astype(int)  # Scores 4 and 5 are considered malignant
junior_cm = confusion_matrix(y_true, junior_binary)
junior_tn, junior_fp, junior_fn, junior_tp = junior_cm.ravel()
junior_accuracy = accuracy_score(y_true, junior_binary)
junior_sensitivity = recall_score(y_true, junior_binary)  # Same as recall
junior_specificity = junior_tn / (junior_tn + junior_fp)

# For accuracy (Junior Expert)
p_junior_acc = junior_accuracy
se_junior_acc = np.sqrt((p_junior_acc * (1 - p_junior_acc)) / n)
ci_95_junior_acc = 1.96 * se_junior_acc
ci_lower_junior_acc = max(0, p_junior_acc - ci_95_junior_acc)
ci_upper_junior_acc = min(1, p_junior_acc + ci_95_junior_acc)

# For sensitivity (Junior Expert)
p_junior_sens = junior_sensitivity
se_junior_sens = np.sqrt((p_junior_sens * (1 - p_junior_sens)) / n_pos) if n_pos > 0 else 0
ci_95_junior_sens = 1.96 * se_junior_sens
ci_lower_junior_sens = max(0, p_junior_sens - ci_95_junior_sens)
ci_upper_junior_sens = min(1, p_junior_sens + ci_95_junior_sens)

# For specificity (Junior Expert)
p_junior_spec = junior_specificity
se_junior_spec = np.sqrt((p_junior_spec * (1 - p_junior_spec)) / n_neg) if n_neg > 0 else 0
ci_95_junior_spec = 1.96 * se_junior_spec
ci_lower_junior_spec = max(0, p_junior_spec - ci_95_junior_spec)
ci_upper_junior_spec = min(1, p_junior_spec + ci_95_junior_spec)

# For Random Forest - use predicted labels from the CSV directly
# This maintains consistency with the original approach
if 'Predicted_Label' in rf_predictions.columns:
    rf_binary = rf_predictions['Predicted_Label'].values
else:
    # If Predicted_Label is not available, fall back to using optimal threshold
    # Calculate optimal threshold from ROC curve
    fpr_rf_opt, tpr_rf_opt, thresholds_rf = roc_curve(y_true, rf_prob_final)
    optimal_idx = np.argmax(tpr_rf_opt - fpr_rf_opt)
    optimal_threshold = thresholds_rf[optimal_idx]
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    rf_binary = (rf_prob_final >= optimal_threshold).astype(int)

rf_cm = confusion_matrix(y_true, rf_binary)
rf_tn, rf_fp, rf_fn, rf_tp = rf_cm.ravel()
rf_accuracy = accuracy_score(y_true, rf_binary)
rf_sensitivity = recall_score(y_true, rf_binary)
rf_specificity = rf_tn / (rf_tn + rf_fp)

# For accuracy (Random Forest)
p_rf_acc = rf_accuracy
se_rf_acc = np.sqrt((p_rf_acc * (1 - p_rf_acc)) / n)
ci_95_rf_acc = 1.96 * se_rf_acc
ci_lower_rf_acc = max(0, p_rf_acc - ci_95_rf_acc)
ci_upper_rf_acc = min(1, p_rf_acc + ci_95_rf_acc)

# For sensitivity (Random Forest)
p_rf_sens = rf_sensitivity
se_rf_sens = np.sqrt((p_rf_sens * (1 - p_rf_sens)) / n_pos) if n_pos > 0 else 0
ci_95_rf_sens = 1.96 * se_rf_sens
ci_lower_rf_sens = max(0, p_rf_sens - ci_95_rf_sens)
ci_upper_rf_sens = min(1, p_rf_sens + ci_95_rf_sens)

# For specificity (Random Forest)
p_rf_spec = rf_specificity
se_rf_spec = np.sqrt((p_rf_spec * (1 - p_rf_spec)) / n_neg) if n_neg > 0 else 0
ci_95_rf_spec = 1.96 * se_rf_spec
ci_lower_rf_spec = max(0, p_rf_spec - ci_95_rf_spec)
ci_upper_rf_spec = min(1, p_rf_spec + ci_95_rf_spec)

# Perform DeLong tests
# Senior Expert vs Random Forest
z_statistic_sr, p_value_sr, delta_auc_sr = delong_test(y_true, senior_expert_scores, rf_prob_final)

# Junior Expert vs Random Forest
z_statistic_jr, p_value_jr, delta_auc_jr = delong_test(y_true, junior_expert_scores, rf_prob_final)

# Senior Expert vs Junior Expert
z_statistic_sj, p_value_sj, delta_auc_sj = delong_test(y_true, senior_expert_scores, junior_expert_scores)

# 定义二分类指标比较函数 - 使用McNemar检验来比较二分类指标
from scipy.stats import chi2_contingency

def compare_binary_metrics(pred1, pred2, y_true):
    """
    使用McNemar检验来比较两个预测模型在二分类任务上的性能差异
    
    参数:
    pred1: 第一个模型的预测标签
    pred2: 第二个模型的预测标签
    y_true: 真实标签
    
    返回:
    chi2值, p值
    """
    # 创建2x2列联表
    # a: 两个模型都预测正确的样本数
    # b: 模型1预测错误，模型2预测正确的样本数
    # c: 模型1预测正确，模型2预测错误的样本数
    # d: 两个模型都预测错误的样本数
    a = np.sum((pred1 == y_true) & (pred2 == y_true))
    b = np.sum((pred1 != y_true) & (pred2 == y_true))
    c = np.sum((pred1 == y_true) & (pred2 != y_true))
    d = np.sum((pred1 != y_true) & (pred2 != y_true))
    
    # 构建列联表
    contingency_table = np.array([[a, b], [c, d]])
    
    # 执行McNemar检验
    result = chi2_contingency(contingency_table, correction=True)
    chi2, p_value = result[:2]
    
    return chi2, p_value

# 计算两个模型预测的相对指标差异
def compute_relative_difference(metric1, metric2):
    """计算两个指标之间的相对差异"""
    return (metric1 - metric2), ((metric1 - metric2) / metric2 * 100)

# 3. 使用McNemar检验比较准确率
# Senior Expert vs Junior Expert
chi2_acc_sj, p_value_acc_sj = compare_binary_metrics(senior_binary, junior_binary, y_true)
acc_diff_sj, acc_rel_diff_sj = compute_relative_difference(senior_accuracy, junior_accuracy)

# Senior Expert vs Random Forest
chi2_acc_sr, p_value_acc_sr = compare_binary_metrics(senior_binary, rf_binary, y_true)
acc_diff_sr, acc_rel_diff_sr = compute_relative_difference(senior_accuracy, rf_accuracy)

# Junior Expert vs Random Forest
chi2_acc_jr, p_value_acc_jr = compare_binary_metrics(junior_binary, rf_binary, y_true)
acc_diff_jr, acc_rel_diff_jr = compute_relative_difference(junior_accuracy, rf_accuracy)

# 4. 针对灵敏度和特异度的比较
# 为灵敏度和特异度创建子集
y_pos = y_true[y_true == 1]  # 阳性样本
y_neg = y_true[y_true == 0]  # 阴性样本

# 对应的预测结果
senior_pos_pred = senior_binary[y_true == 1]
senior_neg_pred = senior_binary[y_true == 0]

junior_pos_pred = junior_binary[y_true == 1]
junior_neg_pred = junior_binary[y_true == 0]

rf_pos_pred = rf_binary[y_true == 1]
rf_neg_pred = rf_binary[y_true == 0]

# 比较灵敏度 (Sensitivity) - 仅使用阳性样本
# Senior Expert vs Junior Expert
chi2_sens_sj, p_value_sens_sj = compare_binary_metrics(senior_pos_pred, junior_pos_pred, y_pos)
sens_diff_sj, sens_rel_diff_sj = compute_relative_difference(senior_sensitivity, junior_sensitivity)

# Senior Expert vs Random Forest
chi2_sens_sr, p_value_sens_sr = compare_binary_metrics(senior_pos_pred, rf_pos_pred, y_pos)
sens_diff_sr, sens_rel_diff_sr = compute_relative_difference(senior_sensitivity, rf_sensitivity)

# Junior Expert vs Random Forest
chi2_sens_jr, p_value_sens_jr = compare_binary_metrics(junior_pos_pred, rf_pos_pred, y_pos)
sens_diff_jr, sens_rel_diff_jr = compute_relative_difference(junior_sensitivity, rf_sensitivity)

# 比较特异度 (Specificity) - 仅使用阴性样本
# 注意：特异度是阴性样本被正确分类为阴性的比例，预测值需要反转
senior_neg_pred_inv = 1 - senior_neg_pred  # 反转预测，使1表示正确分类为阴性
junior_neg_pred_inv = 1 - junior_neg_pred
rf_neg_pred_inv = 1 - rf_neg_pred
y_neg_inv = np.ones_like(y_neg)  # 阴性样本的目标是1（正确分类为阴性）

# Senior Expert vs Junior Expert
chi2_spec_sj, p_value_spec_sj = compare_binary_metrics(senior_neg_pred_inv, junior_neg_pred_inv, y_neg_inv)
spec_diff_sj, spec_rel_diff_sj = compute_relative_difference(senior_specificity, junior_specificity)

# Senior Expert vs Random Forest
chi2_spec_sr, p_value_spec_sr = compare_binary_metrics(senior_neg_pred_inv, rf_neg_pred_inv, y_neg_inv)
spec_diff_sr, spec_rel_diff_sr = compute_relative_difference(senior_specificity, rf_specificity)

# Junior Expert vs Random Forest
chi2_spec_jr, p_value_spec_jr = compare_binary_metrics(junior_neg_pred_inv, rf_neg_pred_inv, y_neg_inv)
spec_diff_jr, spec_rel_diff_jr = compute_relative_difference(junior_specificity, rf_specificity)

# Print results
print(f"Senior Expert AUC: {auc_senior:.4f} (95% CI: [{auc_senior_lower:.4f}, {auc_senior_upper:.4f}])")
print(f"Junior Expert AUC: {auc_junior:.4f} (95% CI: [{auc_junior_lower:.4f}, {auc_junior_upper:.4f}])")
print(f"Random Forest AUC: {auc_rf:.4f} (95% CI: [{auc_rf_lower:.4f}, {auc_rf_upper:.4f}])")

print("\n========== DeLong AUC 比较 ==========")
print(f"Senior Expert vs Junior Expert - Z: {z_statistic_sj:.4f}, p-value: {p_value_sj:.4f}, AUC差异: {delta_auc_sj:.4f}")
print(f"Senior Expert vs Random Forest - Z: {z_statistic_sr:.4f}, p-value: {p_value_sr:.4f}, AUC差异: {delta_auc_sr:.4f}")
print(f"Junior Expert vs Random Forest - Z: {z_statistic_jr:.4f}, p-value: {p_value_jr:.4f}, AUC差异: {delta_auc_jr:.4f}")

print("\n========== 准确率(Accuracy)比较 ==========")
print(f"Senior Expert vs Junior Expert - Chi2: {chi2_acc_sj:.4f}, p-value: {p_value_acc_sj:.4f}")
print(f"  Accuracy差异: {acc_diff_sj:.4f} ({acc_rel_diff_sj:.2f}%)")
print(f"Senior Expert vs Random Forest - Chi2: {chi2_acc_sr:.4f}, p-value: {p_value_acc_sr:.4f}")
print(f"  Accuracy差异: {acc_diff_sr:.4f} ({acc_rel_diff_sr:.2f}%)")
print(f"Junior Expert vs Random Forest - Chi2: {chi2_acc_jr:.4f}, p-value: {p_value_acc_jr:.4f}")
print(f"  Accuracy差异: {acc_diff_jr:.4f} ({acc_rel_diff_jr:.2f}%)")

print("\n========== 灵敏度(Sensitivity)比较 ==========")
print(f"Senior Expert vs Junior Expert - Chi2: {chi2_sens_sj:.4f}, p-value: {p_value_sens_sj:.4f}")
print(f"  Sensitivity差异: {sens_diff_sj:.4f} ({sens_rel_diff_sj:.2f}%)")
print(f"Senior Expert vs Random Forest - Chi2: {chi2_sens_sr:.4f}, p-value: {p_value_sens_sr:.4f}")
print(f"  Sensitivity差异: {sens_diff_sr:.4f} ({sens_rel_diff_sr:.2f}%)")
print(f"Junior Expert vs Random Forest - Chi2: {chi2_sens_jr:.4f}, p-value: {p_value_sens_jr:.4f}")
print(f"  Sensitivity差异: {sens_diff_jr:.4f} ({sens_rel_diff_jr:.2f}%)")

print("\n========== 特异度(Specificity)比较 ==========")
print(f"Senior Expert vs Junior Expert - Chi2: {chi2_spec_sj:.4f}, p-value: {p_value_spec_sj:.4f}")
print(f"  Specificity差异: {spec_diff_sj:.4f} ({spec_rel_diff_sj:.2f}%)")
print(f"Senior Expert vs Random Forest - Chi2: {chi2_spec_sr:.4f}, p-value: {p_value_spec_sr:.4f}")
print(f"  Specificity差异: {spec_diff_sr:.4f} ({spec_rel_diff_sr:.2f}%)")
print(f"Junior Expert vs Random Forest - Chi2: {chi2_spec_jr:.4f}, p-value: {p_value_spec_jr:.4f}")
print(f"  Specificity差异: {spec_diff_jr:.4f} ({spec_rel_diff_jr:.2f}%)")

# Print confusion matrix and metrics for Senior Expert
print("\nSenior Expert Confusion Matrix:")
print(f"TN: {senior_tn}, FP: {senior_fp}")
print(f"FN: {senior_fn}, TP: {senior_tp}")
print(f"Accuracy: {senior_accuracy:.4f} (95% CI: [{ci_lower_senior_acc:.4f}, {ci_upper_senior_acc:.4f}])")
print(f"Sensitivity: {senior_sensitivity:.4f} (95% CI: [{ci_lower_senior_sens:.4f}, {ci_upper_senior_sens:.4f}])")
print(f"Specificity: {senior_specificity:.4f} (95% CI: [{ci_lower_senior_spec:.4f}, {ci_upper_senior_spec:.4f}])")

# Print confusion matrix and metrics for Junior Expert
print("\nJunior Expert Confusion Matrix:")
print(f"TN: {junior_tn}, FP: {junior_fp}")
print(f"FN: {junior_fn}, TP: {junior_tp}")
print(f"Accuracy: {junior_accuracy:.4f} (95% CI: [{ci_lower_junior_acc:.4f}, {ci_upper_junior_acc:.4f}])")
print(f"Sensitivity: {junior_sensitivity:.4f} (95% CI: [{ci_lower_junior_sens:.4f}, {ci_upper_junior_sens:.4f}])")
print(f"Specificity: {junior_specificity:.4f} (95% CI: [{ci_lower_junior_spec:.4f}, {ci_upper_junior_spec:.4f}])")

# Print confusion matrix and metrics for Random Forest
print("\nRandom Forest Confusion Matrix:")
print(f"TN: {rf_tn}, FP: {rf_fp}")
print(f"FN: {rf_fn}, TP: {rf_tp}")
print(f"Accuracy: {rf_accuracy:.4f} (95% CI: [{ci_lower_rf_acc:.4f}, {ci_upper_rf_acc:.4f}])")
print(f"Sensitivity: {rf_sensitivity:.4f} (95% CI: [{ci_lower_rf_sens:.4f}, {ci_upper_rf_sens:.4f}])")
print(f"Specificity: {rf_specificity:.4f} (95% CI: [{ci_lower_rf_spec:.4f}, {ci_upper_rf_spec:.4f}])")

# Plot ROC curves
plt.figure(figsize=(10, 8))
plt.plot(fpr_senior, tpr_senior, 'b-', linewidth=2, label=f'Senior Expert (AUC = {auc_senior:.3f}, 95% CI: [{auc_senior_lower:.3f}-{auc_senior_upper:.3f}])')
plt.plot(fpr_junior, tpr_junior, 'g-', linewidth=2, label=f'Junior Expert (AUC = {auc_junior:.3f}, 95% CI: [{auc_junior_lower:.3f}-{auc_junior_upper:.3f}])')
plt.plot(fpr_rf, tpr_rf, 'r-', linewidth=2, label=f'Random Forest (AUC = {auc_rf:.3f}, 95% CI: [{auc_rf_lower:.3f}-{auc_rf_upper:.3f}])')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves Comparison')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.7)
# plt.savefig('delong_test/roc_comparison_experts_rf.png', dpi=300, bbox_inches='tight')
plt.savefig('sub_delong_test/sub_roc_comparison_experts_rf.png', dpi=300, bbox_inches='tight')
plt.close()
print("ROC curve comparison has been saved as 'roc_comparison_experts_rf.png'")

# 保存更新的比较结果到文件
# with open('delong_test/performance_comparison_results.txt', 'w') as f:
with open('sub_delong_test/sub_performance_comparison_results.txt', 'w') as f:
    f.write("==================== AUC 比较 ====================\n")
    f.write(f"Senior Expert AUC: {auc_senior:.4f} (95% CI: [{auc_senior_lower:.4f}, {auc_senior_upper:.4f}])\n")
    f.write(f"Junior Expert AUC: {auc_junior:.4f} (95% CI: [{auc_junior_lower:.4f}, {auc_junior_upper:.4f}])\n")
    f.write(f"Random Forest AUC: {auc_rf:.4f} (95% CI: [{auc_rf_lower:.4f}, {auc_rf_upper:.4f}])\n\n")
    
    f.write("-------------------- DeLong AUC 显著性测试 --------------------\n")
    f.write(f"DeLong test (Senior vs Junior) - Z: {z_statistic_sj:.4f}, p-value: {p_value_sj:.4f}, AUC差异: {delta_auc_sj:.4f}\n")
    f.write(f"DeLong test (Senior vs RF) - Z: {z_statistic_sr:.4f}, p-value: {p_value_sr:.4f}, AUC差异: {delta_auc_sr:.4f}\n")
    f.write(f"DeLong test (Junior vs RF) - Z: {z_statistic_jr:.4f}, p-value: {p_value_jr:.4f}, AUC差异: {delta_auc_jr:.4f}\n\n")
    
    f.write("==================== 二分类指标比较 ====================\n")
    
    f.write("\n-------------------- 准确率(Accuracy) --------------------\n")
    f.write(f"Senior Expert: {senior_accuracy:.4f} (95% CI: [{ci_lower_senior_acc:.4f}, {ci_upper_senior_acc:.4f}])\n")
    f.write(f"Junior Expert: {junior_accuracy:.4f} (95% CI: [{ci_lower_junior_acc:.4f}, {ci_upper_junior_acc:.4f}])\n")
    f.write(f"Random Forest: {rf_accuracy:.4f} (95% CI: [{ci_lower_rf_acc:.4f}, {ci_upper_rf_acc:.4f}])\n\n")
    
    f.write("McNemar测试结果:\n")
    f.write(f"Senior vs Junior - Chi2: {chi2_acc_sj:.4f}, p-value: {p_value_acc_sj:.4f}\n")
    f.write(f"Senior vs RF - Chi2: {chi2_acc_sr:.4f}, p-value: {p_value_acc_sr:.4f}\n")
    f.write(f"Junior vs RF - Chi2: {chi2_acc_jr:.4f}, p-value: {p_value_acc_jr:.4f}\n\n")
    
    f.write("\n-------------------- 灵敏度(Sensitivity) --------------------\n")
    f.write(f"Senior Expert: {senior_sensitivity:.4f} (95% CI: [{ci_lower_senior_sens:.4f}, {ci_upper_senior_sens:.4f}])\n")
    f.write(f"Junior Expert: {junior_sensitivity:.4f} (95% CI: [{ci_lower_junior_sens:.4f}, {ci_upper_junior_sens:.4f}])\n")
    f.write(f"Random Forest: {rf_sensitivity:.4f} (95% CI: [{ci_lower_rf_sens:.4f}, {ci_upper_rf_sens:.4f}])\n\n")
    
    f.write("McNemar测试结果:\n")
    f.write(f"Senior vs Junior - Chi2: {chi2_sens_sj:.4f}, p-value: {p_value_sens_sj:.4f}\n")
    f.write(f"Senior vs RF - Chi2: {chi2_sens_sr:.4f}, p-value: {p_value_sens_sr:.4f}\n")
    f.write(f"Junior vs RF - Chi2: {chi2_sens_jr:.4f}, p-value: {p_value_sens_jr:.4f}\n\n")
    
    f.write("\n-------------------- 特异度(Specificity) --------------------\n") 
    f.write(f"Senior Expert: {senior_specificity:.4f} (95% CI: [{ci_lower_senior_spec:.4f}, {ci_upper_senior_spec:.4f}])\n")
    f.write(f"Junior Expert: {junior_specificity:.4f} (95% CI: [{ci_lower_junior_spec:.4f}, {ci_upper_junior_spec:.4f}])\n")
    f.write(f"Random Forest: {rf_specificity:.4f} (95% CI: [{ci_lower_rf_spec:.4f}, {ci_upper_rf_spec:.4f}])\n\n")
    
    f.write("McNemar测试结果:\n")
    f.write(f"Senior vs Junior - Chi2: {chi2_spec_sj:.4f}, p-value: {p_value_spec_sj:.4f}\n")
    f.write(f"Senior vs RF - Chi2: {chi2_spec_sr:.4f}, p-value: {p_value_spec_sr:.4f}\n")
    f.write(f"Junior vs RF - Chi2: {chi2_spec_jr:.4f}, p-value: {p_value_spec_jr:.4f}\n\n")
    
    f.write("\n==================== 混淆矩阵 ====================\n")
    f.write("\nSenior Expert Confusion Matrix:\n")
    f.write(f"TN: {senior_tn}, FP: {senior_fp}\n")
    f.write(f"FN: {senior_fn}, TP: {senior_tp}\n")
    
    f.write("\nJunior Expert Confusion Matrix:\n")
    f.write(f"TN: {junior_tn}, FP: {junior_fp}\n")
    f.write(f"FN: {junior_fn}, TP: {junior_tp}\n")
    
    f.write("\nRandom Forest Confusion Matrix:\n")
    f.write(f"TN: {rf_tn}, FP: {rf_fp}\n")
    f.write(f"FN: {rf_fn}, TP: {rf_tp}\n")

print("更新的比较结果已保存到 'performance_comparison_results.txt'")