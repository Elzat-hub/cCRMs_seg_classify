import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import fisher_exact

# 读取Excel文件
df = pd.read_excel('for_P_value.xlsx', sheet_name='Sheet2')

# 计算模型的性能指标
model_accuracy = accuracy_score(df['True_Label'], df['Predicted_Label'])
model_cm = confusion_matrix(df['True_Label'], df['Predicted_Label'])
model_sensitivity = model_cm[1,1] / (model_cm[1,1] + model_cm[1,0])
model_specificity = model_cm[0,0] / (model_cm[0,0] + model_cm[0,1])

# 计算专家的性能指标
expert_accuracy = accuracy_score(df['True_Label'], df['expert'])
expert_cm = confusion_matrix(df['True_Label'], df['expert'])
expert_sensitivity = expert_cm[1,1] / (expert_cm[1,1] + expert_cm[1,0])
expert_specificity = expert_cm[0,0] / (expert_cm[0,0] + expert_cm[0,1])

# 打印结果
print("Model Performance:")
print(f"Accuracy: {model_accuracy:.4f}")
print(f"Sensitivity: {model_sensitivity:.4f}")
print(f"Specificity: {model_specificity:.4f}")

print("\nExpert Performance:")
print(f"Accuracy: {expert_accuracy:.4f}")
print(f"Sensitivity: {expert_sensitivity:.4f}")
print(f"Specificity: {expert_specificity:.4f}")

# 计算P值
accuracy_p = fisher_exact([[sum(df['True_Label'] == df['Predicted_Label']), sum(df['True_Label'] != df['Predicted_Label'])],
                           [sum(df['True_Label'] == df['expert']), sum(df['True_Label'] != df['expert'])]])[1]

sensitivity_p = fisher_exact([[model_cm[1,1], model_cm[1,0]],
                              [expert_cm[1,1], expert_cm[1,0]]])[1]

specificity_p = fisher_exact([[model_cm[0,0], model_cm[0,1]],
                              [expert_cm[0,0], expert_cm[0,1]]])[1]

print("\nP-values:")
print(f"Accuracy P-value: {accuracy_p:.4f}")
print(f"Sensitivity P-value: {sensitivity_p:.4f}")
print(f"Specificity P-value: {specificity_p:.4f}")