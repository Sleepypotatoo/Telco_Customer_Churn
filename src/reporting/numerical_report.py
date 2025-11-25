"""数值特征分析报告（增强版）"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import logging

plt.style.use('seaborn-v0_8')


def _save(fig_name):
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.close()


def numerical_report(df: pd.DataFrame, out_dir: Path = Path('reports')):
    out_dir.mkdir(exist_ok=True)
    md_path = out_dir / 'numerical_feature_report.md'
    num = df.select_dtypes('number')

    md = f"# 数值特征分析报告\n\n样本：{df.shape}\n\n"
    if num.empty:
        md += "> 无数值列  \n"
        md_path.write_text(md, encoding='utf-8')
        return

    # 1. 描述统计 + 偏度峰度变异系数（直接 pandas/scipy）
    from scipy.stats import skew, kurtosis
    stat = num.describe().T
    stat['skew'] = skew(num, nan_policy='omit')
    stat['kurt'] = kurtosis(num, nan_policy='omit')
    stat['cv'] = stat['std'] / stat['mean']
    stat = stat.round(2)
    md += "## 1 描述统计\n" + stat.to_markdown() + "\n\n"

    # 2. 分布图（直接 sns.histplot）
    plt_dir = out_dir / 'plots'
    plt_dir.mkdir(exist_ok=True)
    for col in num.columns[:4]:  # 前 4 列
        plt.figure(figsize=(4, 3))
        sns.histplot(num[col], kde=True)
        plt.title(f'{col} 分布')
        fig = plt_dir / f'dist_{col}.png'
        _save(fig)
        md += f"![{col}]({fig.relative_to(out_dir)}) "

    md += "\n\n"

    # 3. 箱型图（直接 sns.boxplot）
    plt.figure(figsize=(6, 3))
    sns.boxplot(data=num.iloc[:, :4], orient='h')
    plt.title('箱型图（前 4 列）')
    fig_box = plt_dir / 'num_box.png'
    _save(fig_box)
    md += f"![箱型图]({fig_box.relative_to(out_dir)})\n\n"

    # 4. 相关性热力图
    plt.figure(figsize=(6, 5))
    sns.heatmap(num.corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('皮尔逊相关系数')
    fig_corr = plt_dir / 'num_corr.png'
    _save(fig_corr)
    md += f"![相关性]({fig_corr.relative_to(out_dir)})\n\n"

    # 5. 高相关警告（直接 corr）
    high_corr = (
        num.corr()
        .where(np.triu(np.ones(num.shape[1]), k=1).astype(bool))
        .stack()
        .abs()
        .sort_values(ascending=False)
    )
    if high_corr.iloc[0] > 0.8:
        md += f"> ⚠️ 高相关：{high_corr.index[0]} 相关系数 {high_corr.iloc[0]:.2f}，建议降维或删除。  \n\n"
    else:
        md += "> ✅ 无高度相关（>0.8）。  \n\n"

    md_path.write_text(md, encoding='utf-8')
    logging.info(f"[数值报告] Markdown -> {md_path}")