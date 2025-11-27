# 电信客户流失预测 - 特征文档说明

## 数据集概览
- **总样本数**: 7043
- **总特征数**: 35
- **数值型特征**: 8
- **分类型特征**: 0

## 特征详情

### 数值型特征
| 特征名 | 数据类型 | 缺失值 | 唯一值 | 最小值 | 最大值 | 均值 | 标准差 |
|--------|----------|--------|--------|--------|--------|------|--------|
| SeniorCitizen | int64 | 0 | 2 | 0.00 | 1.00 | 0.16 | 0.37 |
| tenure | int64 | 0 | 73 | 0.00 | 72.00 | 32.37 | 24.56 |
| MonthlyCharges | float64 | 0 | 1585 | 18.25 | 118.75 | 64.76 | 30.09 |
| TotalCharges | float64 | 0 | 6531 | 0.00 | 8684.80 | 2279.73 | 2266.79 |
| customer_value | float64 | 0 | 6051 | 0.00 | 8550.00 | 2279.58 | 2264.73 |
| Contract_One year | bool | 0 | 2 | 0.00 | 1.00 | 0.21 | 0.41 |
| Contract_Two year | bool | 0 | 2 | 0.00 | 1.00 | 0.24 | 0.43 |
| Dependents_Yes | bool | 0 | 2 | 0.00 | 1.00 | 0.30 | 0.46 |
| DeviceProtection_No internet service | bool | 0 | 2 | 0.00 | 1.00 | 0.22 | 0.41 |
| DeviceProtection_Yes | bool | 0 | 2 | 0.00 | 1.00 | 0.34 | 0.48 |
| InternetService_Fiber optic | bool | 0 | 2 | 0.00 | 1.00 | 0.44 | 0.50 |
| InternetService_No | bool | 0 | 2 | 0.00 | 1.00 | 0.22 | 0.41 |
| OnlineBackup_No internet service | bool | 0 | 2 | 0.00 | 1.00 | 0.22 | 0.41 |
| OnlineBackup_Yes | bool | 0 | 2 | 0.00 | 1.00 | 0.34 | 0.48 |
| OnlineSecurity_No internet service | bool | 0 | 2 | 0.00 | 1.00 | 0.22 | 0.41 |
| OnlineSecurity_Yes | bool | 0 | 2 | 0.00 | 1.00 | 0.29 | 0.45 |
| PaperlessBilling_Yes | bool | 0 | 2 | 0.00 | 1.00 | 0.59 | 0.49 |
| Partner_Yes | bool | 0 | 2 | 0.00 | 1.00 | 0.48 | 0.50 |
| PaymentMethod_Credit card (automatic) | bool | 0 | 2 | 0.00 | 1.00 | 0.22 | 0.41 |
| PaymentMethod_Electronic check | bool | 0 | 2 | 0.00 | 1.00 | 0.34 | 0.47 |
| PaymentMethod_Mailed check | bool | 0 | 2 | 0.00 | 1.00 | 0.23 | 0.42 |
| StreamingMovies_No internet service | bool | 0 | 2 | 0.00 | 1.00 | 0.22 | 0.41 |
| StreamingMovies_Yes | bool | 0 | 2 | 0.00 | 1.00 | 0.39 | 0.49 |
| StreamingTV_No internet service | bool | 0 | 2 | 0.00 | 1.00 | 0.22 | 0.41 |
| StreamingTV_Yes | bool | 0 | 2 | 0.00 | 1.00 | 0.38 | 0.49 |
| TechSupport_No internet service | bool | 0 | 2 | 0.00 | 1.00 | 0.22 | 0.41 |
| TechSupport_Yes | bool | 0 | 2 | 0.00 | 1.00 | 0.29 | 0.45 |
| contract_numeric_2 | bool | 0 | 2 | 0.00 | 1.00 | 0.21 | 0.41 |
| contract_numeric_3 | bool | 0 | 2 | 0.00 | 1.00 | 0.24 | 0.43 |
| monthly_charges_group_高消费 | bool | 0 | 2 | 0.00 | 1.00 | 0.38 | 0.49 |
| tenure_group_3-5年 | bool | 0 | 2 | 0.00 | 1.00 | 0.23 | 0.42 |
| tenure_group_5年以上 | bool | 0 | 2 | 0.00 | 1.00 | 0.20 | 0.40 |
| monthly_tenure_interaction | float64 | 0 | 6051 | 0.00 | 8550.00 | 2279.58 | 2264.73 |
| avg_monthly_charge | float64 | 0 | 6586 | 0.00 | 121.40 | 64.70 | 30.27 |
| customer_cluster | int32 | 0 | 4 | 0.00 | 3.00 | 1.87 | 1.16 |

## 特征分类说明

### 基础特征
- **原始特征**: 直接从原始数据中提取的特征
- **客户信息特征**: 性别、年龄、合作伙伴、家属等
- **服务使用特征**: 电话服务、多线路、互联网服务等
- **费用特征**: 月费、总费用等

### 衍生特征
- **数值分箱特征**: 在网时长分组、月费分组等
- **交互特征**: 月费与在网时长的交互项等
- **聚合特征**: 服务使用数量、客户价值评分等

### 高级特征
- **聚类特征**: 客户分群
- **PCA特征**: 主成分分析降维特征

## 数据质量
- **整体缺失率**: 0.00%
- **完全缺失特征**: 0 个
- **无缺失特征**: 35 个

*文档自动生成于特征工程完成后*