"""特征选择"""
import pandas as pd
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import logging

def select_rfe(X: pd.DataFrame, y: pd.Series, n_features: int = 15) -> pd.DataFrame:
    """递归特征消除"""
    if n_features >= X.shape[1]:
        return X

    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = RFE(estimator, n_features_to_select=n_features)
    X_sel = selector.fit_transform(X, y)
    mask = selector.support_
    selected = X.columns[mask].tolist()
    logging.info(f"[特征选择] RFE 完成，选出 {len(selected)} 个特征")
    return X[selected]


def select_importance(X: pd.DataFrame, y: pd.Series, threshold: str = 'median') -> pd.DataFrame:
    """基于随机森林特征重要性"""
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = SelectFromModel(estimator, threshold=threshold)
    X_sel = selector.fit_transform(X, y)
    mask = selector.get_support()
    selected = X.columns[mask].tolist()
    logging.info(f"[特征选择] 重要性完成，选出 {len(selected)} 个特征")
    return X[selected]


def select_correlation(df: pd.DataFrame, target_col: str, threshold: float = 0.05) -> pd.DataFrame:
    """皮尔逊相关系数过滤"""
    corr = df.corr(numeric_only=True)[target_col].abs()
    selected = corr[corr > threshold].index.drop(target_col).tolist()
    logging.info(f"[特征选择] 相关性完成，选出 {len(selected)} 个特征")
    return df[selected]


# ---------- 兼容壳 ----------
class FeatureSelector:
    def select_features_rfe(self, X, y, n_features=15):
        return select_rfe(X, y, n_features)

    def select_features_importance(self, X, y, threshold='median'):
        return select_importance(X, y, threshold)

    def select_features_correlation(self, df, target_col, threshold=0.05):
        return select_correlation(df, target_col, threshold)