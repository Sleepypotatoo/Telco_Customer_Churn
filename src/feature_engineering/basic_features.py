"""基础特征工程"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """目标变量 Churn -> Churn_numeric"""
    df = df.copy()
    if 'Churn' in df.columns:
        le = LabelEncoder()
        df['Churn_numeric'] = le.fit_transform(df['Churn'])
        logging.info("[基础特征] 目标变量已编码为 Churn_numeric")
    return df


def bin_numerical(df: pd.DataFrame) -> pd.DataFrame:
    """数值分箱：tenure / MonthlyCharges"""
    df = df.copy()

    if 'tenure' in df.columns:
        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=[0, 12, 24, 36, 60, np.inf],
            labels=['0-1年', '1-2年', '2-3年', '3-5年', '5年以上']
        )

    if 'MonthlyCharges' in df.columns:
        df['monthly_charges_group'] = pd.cut(
            df['MonthlyCharges'],
            bins=[0, 35, 70, 100, np.inf],
            labels=['低消费', '中消费', '高消费', '极高消费']
        )

    logging.info("[基础特征] 数值分箱完成")
    return df


def onehot_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """one-hot 编码分类字段（除 ID 和目标）"""
    cats = df.select_dtypes(['object', 'category']).columns.difference(['customerID', 'Churn'])
    if cats.empty:
        return df

    df = pd.get_dummies(df, columns=cats, drop_first=True)
    logging.info(f"[基础特征] one-hot 完成，新增 {len(df.columns)} 列")
    return df


def create_value_and_service(df: pd.DataFrame) -> pd.DataFrame:
    """客户价值 & 开通服务数量"""
    df = df.copy()

    # 1. 客户价值
    if all(c in df.columns for c in ['MonthlyCharges', 'tenure']):
        df['customer_value'] = df['MonthlyCharges'] * df['tenure']

    # 2. 服务数量
    service_cols = [c for c in df.columns if
                    any(svc in c for svc in
                        ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                         'TechSupport', 'StreamingTV', 'StreamingMovies'])]
    if service_cols:
        # 只统计 0/1 数值列，避免非数值报错
        numeric_svc = [c for c in service_cols if pd.api.types.is_numeric_dtype(df[c])]
        df['num_services'] = df[numeric_svc].sum(axis=1, min_count=1)

    # 3. 合约等级
    if 'Contract' in df.columns:
        mapping = {'Month-to-month': 1, 'One year': 2, 'Two year': 3}
        df['contract_numeric'] = df['Contract'].map(mapping)

    logging.info("[基础特征] 新特征完成（价值/服务数/合约）")
    return df


# ---------- 一键入口 ----------
# 先只做目标编码（不碰其他分类列）
# 再做手工特征（contract_numeric、num_services 等）
# 最后统一 one-hot 剩余所有分类字段
def create_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("[基础特征] 开始基础特征工程")
    df = encode_target(df)
    df = create_value_and_service(df)
    df = bin_numerical(df)
    df = onehot_categorical(df)
    logging.info(f"[基础特征] 完成，当前列数：{df.shape[1]}")
    return df


# ---------- 兼容壳 ----------
class BasicFeatureEngineer:
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return create_basic_features(df)