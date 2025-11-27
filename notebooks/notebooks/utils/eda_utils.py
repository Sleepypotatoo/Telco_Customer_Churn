import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
from IPython.display import display

# =========================================================
# 缺失值分析
# =========================================================
def missing_analysis(df):
    """快速缺失值分析 - 指导清洗策略"""
    print("=== 缺失值分析 ===")
    missing_info = pd.DataFrame({
        '缺失数量': df.isnull().sum(),
        '缺失比例%': (df.isnull().sum() / len(df)) * 100
    }).sort_values('缺失数量', ascending=False)

    # 只显示有缺失的列
    missing_cols = missing_info[missing_info['缺失数量'] > 0]
    print(missing_cols)

    return missing_cols

# =========================================================
# 数据预处理
# =========================================================
def eda_preprocess(df):
    df = df.copy()
    #============= 类型转换 ================
    # 类别处理
    category_cols = [
        'gender', "SeniorCitizen", 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod', 'Churn'
    ]
    category_cols = [c for c in category_cols if c in df.columns]
    df[category_cols] = df[category_cols].astype('category')

    # 数值处理
    if 'TotalCharges' in df.columns:
        # 将字符串转换为数值，空字符串变为 NaN
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    if 'tenure' in df.columns:
        df['tenure'] = df['tenure'].astype('int64')
    float_cols = ['MonthlyCharges', 'TotalCharges']
    float_cols = [c for c in float_cols if c in df.columns]
    df[float_cols] = df[float_cols].astype('float64')

    # ==================== 缺失值分析 ================
    missing_analysis(df)

    #====================== 缺失值处理 ======================
    # 首先处理 TotalCharges 的特殊填充
    if 'TotalCharges' in df.columns:
        na_count = df['TotalCharges'].isna().sum()
        if na_count > 0:
            # 使用 MonthlyCharges * tenure 填充 TotalCharges 的缺失值
            df.loc[df['TotalCharges'].isna(), 'TotalCharges'] = (
                    df.loc[df['TotalCharges'].isna(), 'MonthlyCharges'] *
                    df.loc[df['TotalCharges'].isna(), 'tenure']
            )
            print(f"[清洗] TotalCharges 填充缺失 {na_count} 条")

    # 其他数值列的中位数填充
    num_cols = df.select_dtypes(include=np.number).columns
    # 排除已经处理过的 TotalCharges
    num_cols_to_impute = [col for col in num_cols if col != 'TotalCharges' or df[col].isna().sum() > 0]

    if len(num_cols_to_impute):
        imp = SimpleImputer(strategy='median')
        df[num_cols_to_impute] = imp.fit_transform(df[num_cols_to_impute])
        print(f"[清洗] 数值缺失已用中位数填充：{list(num_cols_to_impute)}")

    # 分类列
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col].fillna(mode_val, inplace=True)
            print(f"[清洗] 分类缺失已用众数填充：{col} -> {mode_val}")

    # ============================ 去重 ================================
    old_rows = len(df)
    df = df.drop_duplicates()
    new_rows = len(df)
    if new_rows < old_rows:
        print(f"[清洗] 去重完成：{old_rows} -> {new_rows}")
    else:
        print("[清洗] 无重复行")

    return df



# =========================================================
# 数据总体概况表
# =========================================================
def dataset_overview(df):
    categorical_cols = ["gender", "SeniorCitizen", "Partner", "Dependents",
                        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "Churn"]
    overview_rows = []

    for col in df.columns:
        col_data = df[col]
        dtype = col_data.dtype
        valid_count = col_data.count()

        # 处理类别特征
        if col in categorical_cols or pd.api.types.is_object_dtype(dtype):
            vc = col_data.value_counts()  # 统计每个类别出现多少次
            top = vc.index[0]  # 出现次数最多的类别名称
            top_ratio = vc.iloc[0] / valid_count if valid_count > 0 else 0  # # 出现次数最多的类别占比

            row = {
                "feature": col,
                "type": "categorical",
                "valid_count": valid_count,
                "min": "-",
                "max": "-",
                "mean": "-",
                "std": "-",
                "skew": "-",
                "detail": f"{len(vc)} 类别, 最常见 {top} ({top_ratio:.1%})"
            }

        # 处理数值特征
        else:
            row = {
                "feature": col,
                "type": "numeric",
                "valid_count": valid_count,
                "min": round(col_data.min(), 2),
                "max": round(col_data.max(), 2),
                "mean": round(col_data.mean(), 2),
                "std": round(col_data.std(), 2),
                "skew": round(col_data.skew(), 2),
                "detail": "数值特征"
            }

        overview_rows.append(row)

    return pd.DataFrame(overview_rows).fillna("")


# =========================================================
# 分布分析
# =========================================================
def plot_distributions(df):
    for col in df.columns:
        if (col == "Churn") :
            continue

        # 为当前图创建一个新的 Matplotlib figure，指定大小为 6 x 4 英寸
        # 这样每个特征都会绘制在单独的图上，避免图像重叠
        plt.figure(figsize=(6, 4))

        # 如果是数值列，绘制直方图并叠加核密度估计曲线 (KDE)
        if pd.api.types.is_numeric_dtype(df[col]):
            sns.histplot(df[col],kde=True)
            plt.title(f"Distribution of {col}")

        # 如果不是数值列（例如 object 或 category），就画类别频数柱状图
        else:
            df[col].value_counts().plot(kind="bar")
            plt.title(f"Category frequencies of {col}")

        # 调整子图布局，避免标题、坐标轴标签或图例重叠
        plt.tight_layout()
        plt.show()


# =========================================================
# 目标变量分析
# =========================================================
def target_analysis(df, target="Churn"):
    plt.figure(figsize=(6, 4))
    df[target].value_counts().plot(kind="bar")
    plt.title(f"Category frequencies of {target}")
    plt.tight_layout()
    plt.show()

