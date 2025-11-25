"""高级特征工程 """
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import logging

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """交互特征：月费×在网时长、平均月费"""
    df = df.copy()
    if all(c in df.columns for c in ['MonthlyCharges', 'tenure']):
        df['monthly_tenure_interaction'] = df['MonthlyCharges'] * df['tenure']

    if all(c in df.columns for c in ['TotalCharges', 'tenure']):
        # 避免除 0
        df['avg_monthly_charge'] = df['TotalCharges'] / np.where(df['tenure'] == 0, 1, df['tenure'])

    logging.info("[高级特征] 交互特征完成")
    return df


def create_cluster_features(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """KMeans 聚类，默认 4 类"""
    df = df.copy()
    cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    avail = [c for c in cols if c in df.columns]
    if len(avail) < 2:
        logging.warning("[高级特征] 聚类所需字段不足，跳过")
        return df

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['customer_cluster'] = kmeans.fit_predict(df[avail])
        logging.info(f"[高级特征] 聚类完成，类别数：{n_clusters}")
    except Exception as e:
        logging.warning(f"[高级特征] 聚类失败：{e}")

    return df


def create_pca_features(df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """PCA 降维，默认保留 2 维"""
    df = df.copy()
    cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'num_services']
    avail = [c for c in cols if c in df.columns]
    if len(avail) < 2:
        logging.warning("[高级特征] PCA 所需字段不足，跳过")
        return df

    try:
        pca = PCA(n_components=n_components, random_state=42)
        pca_result = pca.fit_transform(df[avail])
        for i in range(n_components):
            df[f'pca_{i + 1}'] = pca_result[:, i]
        logging.info(f"[高级特征] PCA 完成，累计解释方差：{pca.explained_variance_ratio_.sum():.3f}")
    except Exception as e:
        logging.warning(f"[高级特征] PCA 失败：{e}")

    return df


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """一键高级特征工程入口"""
    logging.info("[高级特征] 开始高级特征工程")
    df = create_interaction_features(df)
    df = create_cluster_features(df)
    df = create_pca_features(df)
    logging.info("[高级特征] 高级特征工程完成")
    return df


class AdvancedFeatureEngineer:
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return create_advanced_features(df)