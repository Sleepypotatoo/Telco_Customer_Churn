"""EDA 画图函数集合 保留原函数签名，方便主程序无感调用 统一用标准库 logging，不再自建 logger"""
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# 全局样式，一次设定
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

# 统一保存函数，减少重复代码
def _save(fig_path: str, plot_name: str):
    """保存并关闭图，同时写日志"""
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logging.info(f"[EDA] 已保存图表: {plot_name} -> {fig_path}")


# 1. 目标变量分布 -------------------------------------------------------------
def plot_target_distribution(df: pd.DataFrame, save_path: str = None):
    """目标变量 Churn 的饼图+柱状图"""
    if 'Churn' not in df.columns:
        logging.warning("[EDA] 列 Churn 不存在，跳过目标分布图")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 饼图
    df['Churn'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1,
                                        colors=['#2ecc71', '#e74c3c'])
    ax1.set_title('客户流失分布（饼图）')

    # 柱状图
    sns.countplot(x='Churn', data=df, ax=ax2,
                  palette=['#2ecc71', '#e74c3c'])
    ax2.set_title('客户流失分布（柱状图）')
    # 标数字
    for c in ax2.containers:
        ax2.bar_label(c)

    plt.tight_layout()
    if save_path:
        _save(save_path, "目标变量分布图")


# 2. 数值变量分布 -------------------------------------------------------------
def plot_numerical_distributions(df: pd.DataFrame, save_path: str = None):
    """前 4 个数值字段的直方图+密度曲线"""
    nums = df.select_dtypes(include=np.number).columns[:4]
    if nums.empty:
        logging.warning("[EDA] 无数值列，跳过数值分布图")
        return

    n = len(nums)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(nums):
        sns.histplot(data=df, x=col, kde=True, ax=axes[i])
        axes[i].set_title(f'{col} 分布')

    # 隐藏多余子图
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if save_path:
        _save(save_path, "数值变量分布图")


# 3. 分类变量分布 -------------------------------------------------------------
def plot_categorical_distributions(df: pd.DataFrame, save_path: str = None):
    """前 6 个分类字段的条形图（取出现次数前 8 的类别）"""
    cats = [c for c in df.columns if df[c].dtype.name == 'category'
            and c not in {'customerID', 'Churn'}][:6]
    if not cats:
        logging.warning("[EDA] 无分类列，跳过分类分布图")
        return

    cols = 2
    rows = (len(cats) + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if len(cats) > 1 else [axes]

    for i, col in enumerate(cats):
        top8 = df[col].value_counts().head(8).index
        sns.countplot(y=col, data=df[df[col].isin(top8)],
                      order=top8, ax=axes[i])
        axes[i].set_title(f'{col} 分布')
        axes[i].tick_params(axis='y', labelsize=8)

    # 隐藏多余子图
    for j in range(len(cats), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if save_path:
        _save(save_path, "分类变量分布图")


# 4. 相关性热力图 -------------------------------------------------------------
def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None):
    """数值字段皮尔逊相关系数热力图"""
    nums = df.select_dtypes(include=np.number)
    if nums.shape[1] < 2:
        logging.warning("[EDA] 数值列不足 2 个，跳过热力图")
        return

    plt.figure(figsize=(10, 8))
    corr = nums.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', square=True, linewidths=.5,
                cbar_kws={"shrink": .8})
    plt.title('数值特征相关性热力图')
    if save_path:
        _save(save_path, "相关性热力图")


# 5. 按特征统计流失率 ---------------------------------------------------------
def plot_churn_rates_by_features(df: pd.DataFrame, save_path: str = None):
    """看 Contract / InternetService / PaymentMethod 的流失率"""
    feats = ['Contract', 'InternetService', 'PaymentMethod']
    feats = [f for f in feats if f in df.columns]
    if not feats:
        logging.warning("[EDA] 无指定字段，跳过流失率柱状图")
        return

    n = len(feats)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    axes = axes if n > 1 else [axes]

    for ax, feat in zip(axes, feats):
        # 计算流失率
        rate = (df.groupby(feat)['Churn']
                  .apply(lambda x: (x == 'Yes').mean() * 100))
        rate.plot.bar(ax=ax, color='#e74c3c', rot=45)
        ax.set_title(f'{feat} 流失率')
        ax.set_ylabel('流失率 (%)')

    plt.tight_layout()
    if save_path:
        _save(save_path, "特征流失率图")


# 6. 在网时长 vs 流失 ---------------------------------------------------------
def plot_tenure_vs_churn(df: pd.DataFrame, save_path: str = None):
    """Tenure 按流失分组箱线图"""
    if 'tenure' not in df.columns or 'Churn' not in df.columns:
        logging.warning("[EDA] 缺少 tenure 或 Churn，跳过在网时长箱线图")
        return

    plt.figure(figsize=(6, 5))
    sns.boxplot(x='Churn', y='tenure', data=df,
                palette=['#2ecc71', '#e74c3c'])
    plt.title('在网时长 vs 流失')
    if save_path:
        _save(save_path, "在网时长箱线图")


# 7. 费用 vs 流失 -------------------------------------------------------------
def plot_charges_vs_churn(df: pd.DataFrame, save_path: str = None):
    """MonthlyCharges  vs  TotalCharges 散点图，按流失着色"""
    needed = {'MonthlyCharges', 'TotalCharges', 'Churn'}
    if not needed.issubset(df.columns):
        logging.warning("[EDA] 缺少费用字段，跳过费用散点图")
        return

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x='MonthlyCharges', y='TotalCharges',
                    hue='Churn', alpha=0.7,
                    palette=['#2ecc71', '#e74c3c'])
    plt.title('月费 vs 总费用（按流失着色）')
    if save_path:
        _save(save_path, "费用散点图")


# 8. 服务开通情况 -------------------------------------------------------------
def plot_services_usage(df: pd.DataFrame, save_path: str = None):
    """电话/网络/附加服务开通比例条形图"""
    services = ['PhoneService', 'InternetService', 'StreamingTV']
    services = [s for s in services if s in df.columns]
    if not services:
        logging.warning("[EDA] 无服务字段，跳过服务开通图")
        return

    n = len(services)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    axes = axes if n > 1 else [axes]

    for ax, serv in zip(axes, services):
        (df[serv]
         .value_counts(normalize=True)
         .mul(100)
         .plot.bar(ax=ax, color='#3498db', rot=0))
        ax.set_title(f'{serv} 开通比例')
        ax.set_ylabel('百分比 (%)')

    plt.tight_layout()
    if save_path:
        _save(save_path, "服务开通比例图")


# ---------------------------------------------------------------------------
# 为了保持原调用方式，仍提供一个“壳”类，里面全是静态方法
# 内部直接调用上面写好的纯函数，无冗余逻辑
# ---------------------------------------------------------------------------
class EDAPlots:
    """兼容原项目接口的壳，逻辑已全部挪到外部函数，方便后续彻底删除"""
    def __init__(self, figsize=(10, 6)):
        # 仅保留参数，实际不再使用
        self.figsize = figsize

    def plot_target_distribution(self, df, save_path=None):
        plot_target_distribution(df, save_path)

    def plot_numerical_distributions(self, df, save_path=None):
        plot_numerical_distributions(df, save_path)

    def plot_categorical_distributions(self, df, save_path=None):
        plot_categorical_distributions(df, save_path)

    def plot_correlation_heatmap(self, df, save_path=None):
        plot_correlation_heatmap(df, save_path)

    def plot_churn_rates_by_features(self, df, save_path=None):
        plot_churn_rates_by_features(df, save_path)

    def plot_tenure_vs_churn(self, df, save_path=None):
        plot_tenure_vs_churn(df, save_path)

    def plot_charges_vs_churn(self, df, save_path=None):
        plot_charges_vs_churn(df, save_path)

    def plot_services_usage(self, df, save_path=None):
        plot_services_usage(df, save_path)