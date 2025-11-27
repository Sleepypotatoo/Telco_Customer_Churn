"""
数值特征报告 
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import skew, kurtosis
import logging

plt.style.use('seaborn-v0_8')


def numerical_report(df: pd.DataFrame, out_dir: Path = Path('reports')):
    out_dir.mkdir(exist_ok=True)
    md_path = out_dir / 'numerical_feature_report.md'
    num = df.select_dtypes('number')

    md = f"# 数值特征报告\n\n样本：{df.shape}\n\n"
    if num.empty:
        md += "> 无数值列\n"
        md_path.write_text(md, encoding='utf-8')
        return

    # 1. 描述 + 偏度 + 峰度 + CV
    stat = num.describe().T
    stat['skew'] = skew(num, nan_policy='omit')
    stat['kurt'] = kurtosis(num, nan_policy='omit')
    stat['cv'] = stat['std'] / stat['mean']
    md += "## 1 描述统计\n" + stat.round(2).to_markdown() + "\n\n"

    # 2. 分布图（前 4 列）
    plt_dir = out_dir / 'plots'
    plt_dir.mkdir(exist_ok=True)
    for col in num.columns[:4]:
        plt.figure(figsize=(4, 2))
        sns.histplot(num[col], kde=True)
        plt.title(f'{col} 分布')
        fig = plt_dir / f'dist_{col}.png'
        plt.savefig(fig, dpi=300); plt.close()
        md += f"![{col}]({fig.relative_to(out_dir)}) "

    md += "\n\n"

    # 3. 箱型图
    plt.figure(figsize=(6, 3))
    sns.boxplot(data=num.iloc[:, :4], orient='h')
    plt.title('箱型图（前 4 列）')
    fig_box = plt_dir / 'num_box.png'
    plt.savefig(fig_box, dpi=300); plt.close()
    md += f"![箱型图]({fig_box.relative_to(out_dir)})\n\n"

    # 4. 联合图（tenure vs MonthlyCharges）
    if {'tenure', 'MonthlyCharges'}.issubset(num.columns):
        plt.figure(figsize=(4, 3))
        sns.jointplot(x='tenure', y='MonthlyCharges', data=num, kind='scatter', alpha=0.6)
        plt.tight_layout()
        fig_joint = plt_dir / 'tenure_vs_monthly.png'
        plt.savefig(fig_joint, dpi=300); plt.close()
        md += f"![联合图]({fig_joint.relative_to(out_dir)})\n\n"

    # 5. 相关性热力图
    plt.figure(figsize=(6, 5))
    sns.heatmap(num.corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('皮尔逊相关系数')
    fig_corr = plt_dir / 'num_corr.png'
    plt.savefig(fig_corr, dpi=300); plt.close()
    md += f"![相关性]({fig_corr.relative_to(out_dir)})\n\n"

    # 6. 高相关警告
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
    logging.info(f"[数值报告] 报告 -> {md_path}")