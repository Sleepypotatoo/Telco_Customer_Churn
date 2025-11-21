# src/data_processing/eda.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class EDA:
    """
    探索性数据分析类
    """
    
    def __init__(self):
        self.eda_report = {}
    
    def perform_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        执行完整的探索性数据分析
        
        Args:
            df: 输入数据框
            
        Returns:
            Dict[str, Any]: EDA报告
        """
        logger.info("开始探索性数据分析")
        
        self.eda_report = {}
        
        # 基本数据信息
        self._basic_info(df)
        
        # 数值变量分析
        self._numerical_analysis(df)
        
        # 分类变量分析
        self._categorical_analysis(df)
        
        # 缺失值分析
        self._missing_value_analysis(df)
        
        # 相关性分析
        self._correlation_analysis(df)
        
        # 目标变量分析
        self._target_analysis(df)
        
        logger.info("探索性数据分析完成")
        
        return self.eda_report
    
    def _basic_info(self, df: pd.DataFrame):
        """基本数据信息"""
        self.eda_report['basic_info'] = {
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.astype(str).to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        logger.info(f"数据形状: {df.shape}")
        logger.info(f"内存使用: {self.eda_report['basic_info']['memory_usage'] / 1024 / 1024:.2f} MB")
    
    def _numerical_analysis(self, df: pd.DataFrame):
        """数值变量分析"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            numerical_stats = df[numerical_cols].describe().to_dict()
            self.eda_report['numerical_stats'] = numerical_stats
            
            logger.info(f"分析 {len(numerical_cols)} 个数值变量")
    
    def _categorical_analysis(self, df: pd.DataFrame):
        """分类变量分析"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        categorical_stats = {}
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            categorical_stats[col] = {
                'unique_count': len(value_counts),
                'top_categories': value_counts.head(10).to_dict()
            }
        
        self.eda_report['categorical_stats'] = categorical_stats
        
        logger.info(f"分析 {len(categorical_cols)} 个分类变量")
    
    def _missing_value_analysis(self, df: pd.DataFrame):
        """缺失值分析"""
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        missing_info = pd.DataFrame({
            'missing_count': missing_data,
            'missing_percent': missing_percent
        })
        
        # 只保留有缺失值的列
        missing_info = missing_info[missing_info['missing_count'] > 0]
        
        self.eda_report['missing_values'] = missing_info.to_dict()
        
        if len(missing_info) > 0:
            logger.info(f"发现 {len(missing_info)} 列有缺失值")
        else:
            logger.info("未发现缺失值")
    
    def _correlation_analysis(self, df: pd.DataFrame):
        """相关性分析"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 1:
            correlation_matrix = df[numerical_cols].corr()
            self.eda_report['correlation_matrix'] = correlation_matrix.to_dict()
            
            # 找出高相关性特征对
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if abs(correlation_matrix.iloc[i, j]) > 0.5:
                        high_corr_pairs.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': correlation_matrix.iloc[i, j]
                        })
            
            self.eda_report['high_correlation_pairs'] = high_corr_pairs
            
            logger.info(f"发现 {len(high_corr_pairs)} 对高相关性特征")
    
    def _target_analysis(self, df: pd.DataFrame):
        """目标变量分析"""
        if 'Churn' in df.columns:
            churn_distribution = df['Churn'].value_counts()
            churn_rate = (churn_distribution / len(df)) * 100
            
            self.eda_report['target_analysis'] = {
                'distribution': churn_distribution.to_dict(),
                'rates': churn_rate.to_dict()
            }
            
            logger.info(f"流失率: {churn_rate.get('Yes', 0):.2f}%")
    
    def get_summary(self) -> str:
        """获取EDA摘要"""
        if not self.eda_report:
            return "尚未执行EDA分析"
        
        summary = []
        basic_info = self.eda_report.get('basic_info', {})
        summary.append(f"数据形状: {basic_info.get('shape', 'N/A')}")
        summary.append(f"列数: {len(basic_info.get('columns', []))}")
        
        if 'target_analysis' in self.eda_report:
            rates = self.eda_report['target_analysis'].get('rates', {})
            churn_rate = rates.get('Yes', 0)
            summary.append(f"流失率: {churn_rate:.2f}%")
        
        return "\n".join(summary)