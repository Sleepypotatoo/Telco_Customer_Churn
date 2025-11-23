# src/visualization/eda_plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# 设置中文字体（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

class EDAPlots:
    """
    EDA可视化类
    """
    
    def __init__(self, figsize=(10, 6)):
        self.figsize = figsize
    
    def plot_target_distribution(self, df: pd.DataFrame, save_path: str = None):
        """绘制目标变量分布"""
        if 'Churn' not in df.columns:
            logger.warning("数据中未找到Churn列")
            return
        
        plt.figure(figsize=self.figsize)
        
        # 计算分布
        churn_counts = df['Churn'].value_counts()
        colors = ['#2ecc71', '#e74c3c']  # 绿色表示未流失，红色表示流失
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 饼图
        ax1.pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax1.set_title('Customer Churn Distribution - Pie Chart')
        
        # 柱状图
        bars = ax2.bar(churn_counts.index, churn_counts.values, color=colors, alpha=0.7)
        ax2.set_title('Customer Churn Distribution - Bar Chart')
        ax2.set_xlabel('Churn')
        ax2.set_ylabel('Customer Count')
        
        # 在柱状图上添加数值
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"目标变量分布图已保存: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_numerical_distributions(self, df: pd.DataFrame, save_path: str = None):
        """绘制数值变量分布"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            logger.warning("未找到数值变量")
            return
        
        # 选择前6个数值变量进行展示
        cols_to_plot = numerical_cols[:6]
        n_cols = min(3, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(cols_to_plot):
            if i < len(axes):
                axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_title(f'{col} Distribution')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # 隐藏多余的子图
        for i in range(len(cols_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"数值变量分布图已保存: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_categorical_distributions(self, df: pd.DataFrame, save_path: str = None):
        """绘制分类变量分布"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # 排除ID列和目标列
        cols_to_exclude = ['customerID', 'Churn']
        categorical_cols = [col for col in categorical_cols if col not in cols_to_exclude]
        
        if len(categorical_cols) == 0:
            logger.warning("未找到分类变量")
            return
        
        # 选择前6个分类变量进行展示
        cols_to_plot = categorical_cols[:6]
        n_cols = min(3, len(cols_to_plot))
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(cols_to_plot):
            if i < len(axes):
                value_counts = df[col].value_counts().head(10)  # 只显示前10个类别
                bars = axes[i].bar(range(len(value_counts)), value_counts.values, 
                                 color=plt.cm.Set3(np.linspace(0, 1, len(value_counts))))
                axes[i].set_title(f'{col} Distribution')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                axes[i].set_xticks(range(len(value_counts)))
                axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
        
        # 隐藏多余的子图
        for i in range(len(cols_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"分类变量分布图已保存: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, save_path: str = None):
        """绘制相关性热力图"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            logger.warning("数值变量数量不足，无法绘制相关性热力图")
            return
        
        plt.figure(figsize=(10, 8))
        
        correlation_matrix = df[numerical_cols].corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, fmt='.2f')
        plt.title('Numerical Features Correlation Heatmap')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"相关性热力图已保存: {save_path}")
        
        plt.show()
        plt.close()