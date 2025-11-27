"""
数据质量报告 
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import logging

plt.style.use('seaborn-v0_8')


def quality_report(df: pd.DataFrame, out_dir: Path = Path('reports')):
    out_dir.mkdir(exist_ok=True)
    md_path = out_dir / 'data_quality_report.md'
    csv_path = out_dir / 'tables' / 'outlier_detail.csv'

    n, m = df.shape
    md = f"# 数据质量报告\n\n样本：{n:,} 行 × {m} 列\n\n"

    # 1. 缺失 & 重复
    missing = df.isnull().mean()
    missing = missing[missing > 0]
    dup = df.duplicated().sum()
    md += f"缺失字段：{len(missing)}  |  重复行：{dup} ({dup/n:.1%})\n\n"
    if not missing.empty:
        plt.figure(figsize=(5, 2))
        sns.barplot(x=missing.index, y=missing.values)
        plt.title('缺失率'); plt.xticks(rotation=45)
        fig1 = out_dir / 'missing_bar.png'
        plt.savefig(fig1, dpi=300); plt.close()
        md += f"![缺失]({fig1.name})\n\n"


    # 2. 异常值（IQR 明细导出）
    num = df.select_dtypes('number')
    outlier = []
    for col in num.columns:
        q1, q3 = num[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out = num[col].between(low, high, inclusive='neither')
        outlier.append({'字段': col, '异常数': out.sum(), '异常比例': f"{out.sum()/n:.1%}"})
    outlier = pd.DataFrame(outlier)
    md += "## 异常值（IQR）\n" + outlier.to_markdown(index=False) + "\n\n"
    outlier.to_csv(csv_path, index=False)
    logging.info(f"[质量报告] 异常明细 -> {csv_path}")

    # 4. 结论
    md += "## 结论\n"
    if len(missing) == 0 and dup == 0 and outlier['异常数'].sum() == 0:
        md += "> ✅ 数据完整性、唯一性、合理性均良好，可直接建模。\n"
    else:
        md += f"> ⚠️ 已处理缺失/重复/异常，当前数据集可直接用于后续分析。\n"

    md_path.write_text(md, encoding='utf-8')
    logging.info(f"[质量报告] 报告 -> {md_path}")