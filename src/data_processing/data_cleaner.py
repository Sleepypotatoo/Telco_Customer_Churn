import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import logging

# 全局日志配置（只配置一次，由主程序统一控制格式）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# 1. 处理 TotalCharges -------------------------------------------------------
def handle_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """TotalCharges 转数值 + 缺失用月费*在网时长填充"""
    df = df.copy()
    # 转数值，非法变 NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    na_count = df['TotalCharges'].isna().sum()
    if na_count > 0:
        df.loc[df['TotalCharges'].isna(), 'TotalCharges'] = (
            df.loc[df['TotalCharges'].isna(), 'MonthlyCharges'] *
            df.loc[df['TotalCharges'].isna(), 'tenure']
        )
        logging.info(f"[清洗] TotalCharges 填充缺失 {na_count} 条")
    return df


# 2. 缺失值处理 --------------------------------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """数值列用中位数，分类列用众数"""
    df = df.copy()

    # 数值列
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols):
        imp = SimpleImputer(strategy='median')
        df[num_cols] = imp.fit_transform(df[num_cols])
        logging.info(f"[清洗] 数值缺失已用中位数填充：{list(num_cols)}")

    # 分类列
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col].fillna(mode_val, inplace=True)
            logging.info(f"[清洗] 分类缺失已用众数填充：{col} -> {mode_val}")

    return df


# 3. 类型转换 ----------------------------------------------------------------
def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """按需把列转 category / int / float"""
    df = df.copy()

    category_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod', 'Churn'
    ]
    # 只保留存在的列
    category_cols = [c for c in category_cols if c in df.columns]
    df[category_cols] = df[category_cols].astype('category')

    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].astype('int64')
    if 'tenure' in df.columns:
        df['tenure'] = df['tenure'].astype('int64')

    float_cols = ['MonthlyCharges', 'TotalCharges']
    float_cols = [c for c in float_cols if c in df.columns]
    df[float_cols] = df[float_cols].astype('float64')

    logging.info(f"[清洗] 类型转换完成")
    return df


# 4. 去重 --------------------------------------------------------------------
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """物理去重"""
    old_rows = len(df)
    df = df.drop_duplicates()
    new_rows = len(df)
    if new_rows < old_rows:
        logging.info(f"[清洗] 去重完成：{old_rows} -> {new_rows}")
    else:
        logging.info("[清洗] 无重复行")
    return df


# 5. 一键清洗入口 ------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """顺序执行所有清洗步骤"""
    logging.info("[清洗] 开始数据清洗")
    df = handle_total_charges(df)
    df = handle_missing_values(df)
    df = convert_data_types(df)
    df = remove_duplicates(df)
    logging.info(f"[清洗] 清洗完成，最终形状：{df.shape}")
    return df


# 6. 兼容原接口的壳（可选，主程序已改函数调用时可删除） ------------------------
class DataCleaner:
    """壳，保持原 new DataCleaner().clean_data(df) 调用方式"""
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return clean_data(df)

    def print_cleaning_summary(self):
        """原方法留空，避免主程序报错"""
        print("=" * 50)
        print("数据清洗摘要（纯函数版）")
        print("=" * 50)
        print("详见日志输出，不再重复打印")
        print("=" * 50)