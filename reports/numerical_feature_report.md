# 数值特征分析报告

样本：(7043, 21)

## 1 描述统计
|                |   count |    mean |     std |   min |    25% |     50% |     75% |     max |   skew |   kurt |   cv |
|:---------------|--------:|--------:|--------:|------:|-------:|--------:|--------:|--------:|-------:|-------:|-----:|
| SeniorCitizen  |    7043 |    0.16 |    0.37 |  0    |   0    |    0    |    0    |    1    |   1.83 |   1.36 | 2.27 |
| tenure         |    7043 |   32.37 |   24.56 |  0    |   9    |   29    |   55    |   72    |   0.24 |  -1.39 | 0.76 |
| MonthlyCharges |    7043 |   64.76 |   30.09 | 18.25 |  35.5  |   70.35 |   89.85 |  118.75 |  -0.22 |  -1.26 | 0.46 |
| TotalCharges   |    7043 | 2279.73 | 2266.79 |  0    | 398.55 | 1394.55 | 3786.6  | 8684.8  |   0.96 |  -0.23 | 0.99 |

![SeniorCitizen](plots\dist_SeniorCitizen.png) ![tenure](plots\dist_tenure.png) ![MonthlyCharges](plots\dist_MonthlyCharges.png) ![TotalCharges](plots\dist_TotalCharges.png) 

![箱型图](plots\num_box.png)

![相关性](plots\num_corr.png)

> ⚠️ 高相关：('tenure', 'TotalCharges') 相关系数 0.83，建议降维或删除。  

