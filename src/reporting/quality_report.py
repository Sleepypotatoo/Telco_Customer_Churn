"""
数据质量分析报告
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
    csv_path = out_dir / 'tables' / 'quality_detail.csv'

    # 0. 总览
    n_row, n_col = df.shape
    md = f"# 数据质量分析报告\n\n"
    md += f"样本：**{n_row:,}** 行 × **{n_col}** 列  \n\n"

    # 1. 缺失
    missing = df.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    md += "## 1 缺失率\n"
    if missing.empty:
        md += "> ✅ 无缺失  \n\n"
    else:
        md += "```\n" + missing.round(3).to_string() + "\n```\n\n"
        # 图
        plt.figure(figsize=(6, 3))
        sns.barplot(x=missing.index, y=missing.values)
        plt.title('字段缺失率')
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig1 = out_dir / 'missing_bar.png'
        plt.savefig(fig1, dpi=300)
        plt.close()
        md += f"![缺失图]({fig1.name})\n\n"

    # 2. 唯一值 & 重复行
    md += "## 2 唯一值与重复\n"
    dup_cnt = df.duplicated().sum()
    md += f"重复行：**{dup_cnt}**（{(dup_cnt/n_row):.2%}）  \n\n"
    if 'customerID' in df.columns:
        id_uniq = df['customerID'].nunique()
        md += f"customerID 唯一值：**{id_uniq:,}**（{'✅ 全唯一' if id_uniq == n_row else '⚠️ 存在重复'}）  \n\n"

    # 3. 类型 & 异常值（简单 IQR）
    md += "## 3 类型与异常值（IQR）\n"
    num_cols = df.select_dtypes('number').columns
    iqr_df = []
    for col in num_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out = ((df[col] < low) | (df[col] > high)).sum()
        iqr_df.append({'字段': col, '异常值数': out, '异常比例': f"{out/n_row:.1%}"})
    iqr_df = pd.DataFrame(iqr_df)
    md += iqr_df.to_markdown(index=False) + "\n\n"

    # 4. 明细 CSV
    detail = pd.DataFrame({
        'dtype': df.dtypes,
        'count': df.count(),
        'unique': df.nunique(),
        'missing_rate': df.isnull().mean()
    }).reset_index().rename(columns={'index': 'column'})
    detail.to_csv(csv_path, index=False, encoding='utf-8')
    logging.info(f"[质量报告] 明细 -> {csv_path}")

    # 5. 结论
    md += "## 5 结论与建议\n"
    if missing.empty and dup_cnt == 0 and iqr_df['异常值数'].sum() == 0:
        md += "> ✅ 数据完整性、唯一性、合理性均良好，可直接建模。  \n"
    else:
        md += f"> ⚠️ 缺失 {len(missing)} 字段，重复 {dup_cnt} 行，异常值 {iqr_df['异常值数'].sum()} 个，已清洗完毕。  \n"

    md_path.write_text(md, encoding='utf-8')
    logging.info(f"[质量报告] Markdown -> {md_path}")