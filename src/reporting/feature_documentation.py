# src/reporting/feature_documentation.py
import pandas as pd
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def generate_feature_documentation(df: pd.DataFrame, save_path: str = "reports/feature_documentation.md"):
    """
    生成特征文档说明
    Args:df: 特征工程后的数据框
    """
    logger.info("生成特征文档说明")
    
    # 确保目录存在
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 分析特征信息
    feature_info = analyze_features(df)
    
    # 生成Markdown文档
    md_content = create_markdown_document(feature_info, df)
    
    # 保存文档
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    logger.info(f"特征文档已保存: {save_path}")
    return md_content

def analyze_features(df: pd.DataFrame) -> dict:
    """分析特征信息"""
    feature_info = {
        'basic_info': {
            'total_features': len(df.columns),
            'total_samples': len(df),
            'numerical_features': len(df.select_dtypes(include=['number']).columns),
            'categorical_features': len(df.select_dtypes(include=['object', 'category']).columns)
        },
        'features': {}
    }
    
    for col in df.columns:
        col_info = {
            'dtype': str(df[col].dtype),
            'missing_count': df[col].isnull().sum(),
            'missing_percent': (df[col].isnull().sum() / len(df)) * 100,
            'unique_count': df[col].nunique()
        }
        
        # 数值型特征统计
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std()
            })
        # 分类型特征统计
        else:
            top_values = df[col].value_counts().head(5).to_dict()
            col_info['top_values'] = top_values
        
        feature_info['features'][col] = col_info
    
    return feature_info

def create_markdown_document(feature_info: dict, df: pd.DataFrame) -> str:
    """创建Markdown格式的文档"""
    content = [
        "# 电信客户流失预测 - 特征文档说明",
        "",
        "## 数据集概览",
        f"- **总样本数**: {feature_info['basic_info']['total_samples']}",
        f"- **总特征数**: {feature_info['basic_info']['total_features']}",
        f"- **数值型特征**: {feature_info['basic_info']['numerical_features']}",
        f"- **分类型特征**: {feature_info['basic_info']['categorical_features']}",
        "",
        "## 特征详情",
        ""
    ]
    
    # 数值型特征表格
    numerical_features = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if numerical_features:
        content.extend([
            "### 数值型特征",
            "| 特征名 | 数据类型 | 缺失值 | 唯一值 | 最小值 | 最大值 | 均值 | 标准差 |",
            "|--------|----------|--------|--------|--------|--------|------|--------|"
        ])
        
        for col in numerical_features:
            info = feature_info['features'][col]
            row = f"| {col} | {info['dtype']} | {info['missing_count']} | {info['unique_count']} | {info['min']:.2f} | {info['max']:.2f} | {info['mean']:.2f} | {info['std']:.2f} |"
            content.append(row)
        content.append("")
    
    # 分类型特征表格
    categorical_features = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    if categorical_features:
        content.extend([
            "### 分类型特征",
            "| 特征名 | 数据类型 | 缺失值 | 唯一值 | 主要取值 |",
            "|--------|----------|--------|--------|----------|"
        ])
        
        for col in categorical_features:
            info = feature_info['features'][col]
            top_values = ", ".join([f"{k}({v})" for k, v in info.get('top_values', {}).items()][:3])
            row = f"| {col} | {info['dtype']} | {info['missing_count']} | {info['unique_count']} | {top_values} |"
            content.append(row)
        content.append("")
    
    # 特征分类说明
    content.extend([
        "## 特征分类说明",
        "",
        "### 基础特征",
        "- **原始特征**: 直接从原始数据中提取的特征",
        "- **客户信息特征**: 性别、年龄、合作伙伴、家属等",
        "- **服务使用特征**: 电话服务、多线路、互联网服务等",
        "- **费用特征**: 月费、总费用等",
        "",
        "### 衍生特征",
        "- **数值分箱特征**: 在网时长分组、月费分组等",
        "- **交互特征**: 月费与在网时长的交互项等",
        "- **聚合特征**: 服务使用数量、客户价值评分等",
        "",
        "### 高级特征", 
        "- **聚类特征**: 客户分群",
        "- **PCA特征**: 主成分分析降维特征",
        "",
        "## 数据质量",
        f"- **整体缺失率**: {(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.2f}%",
        f"- **完全缺失特征**: {sum(df.isnull().sum() == len(df))} 个",
        f"- **无缺失特征**: {sum(df.isnull().sum() == 0)} 个",
        "",
        "*文档自动生成于特征工程完成后*"
    ])
    
    return "\n".join(content)

def save_feature_info_json(feature_info: dict, save_path: str = "reports/feature_info.json"):
    """保存特征信息为JSON格式"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(feature_info, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"特征信息JSON已保存: {save_path}")