# src/data_processing/data_cleaner.py
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any

# 设置日志
logger = logging.getLogger(__name__)

class DataCleaner:
    """
    数据清洗类
    负责处理缺失值、异常值、数据类型转换等数据清洗任务
    """
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化数据清洗器 
        Args:config: 配置参数字典
        """
        self.config = config or {}
        self.cleaning_report = {}
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        主数据清洗函数
        Args:df: 原始数据框    
        Returns:pd.DataFrame: 清洗后的数据框
        """
        logger.info("开始数据清洗过程")
        df_clean = df.copy()
        
        # 记录原始数据信息
        self.cleaning_report['original_shape'] = df_clean.shape
        self.cleaning_report['original_columns'] = list(df_clean.columns)
        
        # 数据清洗步骤
        df_clean = self._handle_total_charges(df_clean)
        df_clean = self._handle_missing_values(df_clean)
        df_clean = self._convert_data_types(df_clean)
        df_clean = self._remove_duplicates(df_clean)
        df_clean = self._handle_outliers(df_clean)
        
        # 记录清洗后数据信息
        self.cleaning_report['cleaned_shape'] = df_clean.shape
        self.cleaning_report['cleaned_columns'] = list(df_clean.columns)
        
        logger.info(f"数据清洗完成: {self.cleaning_report['original_shape']} -> {self.cleaning_report['cleaned_shape']}")
        
        return df_clean
    
    def _handle_total_charges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理TotalCharges列的特殊问题
        Args:df: 输入数据框    
        Returns:pd.DataFrame: 处理后的数据框
        """
        logger.info("处理TotalCharges列")
        
        # 将TotalCharges转换为数值类型，无法转换的设为NaN
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # 记录转换情况
        converted_count = df['TotalCharges'].isna().sum()
        logger.info(f"TotalCharges转换: {converted_count}个值无法转换，设为NaN")
        
        self.cleaning_report['total_charges_converted'] = converted_count
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理缺失值
        Args:df: 输入数据框    
        Returns:pd.DataFrame: 处理缺失值后的数据框
        """
        logger.info("处理缺失值")
        
        # 计算缺失值统计
        missing_stats = df.isnull().sum()
        missing_percent = (missing_stats / len(df)) * 100
        
        missing_info = pd.DataFrame({
            'missing_count': missing_stats,
            'missing_percent': missing_percent
        })
        
        # 只保留有缺失值的列
        missing_info = missing_info[missing_info['missing_count'] > 0]
        
        logger.info(f"发现 {len(missing_info)} 列有缺失值")
        
        # 处理TotalCharges的缺失值 - 用月费*在网时长填充
        if 'TotalCharges' in df.columns and df['TotalCharges'].isna().any():
            mask = df['TotalCharges'].isna()
            df.loc[mask, 'TotalCharges'] = (
                df.loc[mask, 'MonthlyCharges'] * df.loc[mask, 'tenure']
            )
            logger.info(f"填充了 {mask.sum()} 个TotalCharges缺失值")
        
        # 处理其他数值列的缺失值 - 用中位数填充
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"列 '{col}' 用中位数 {median_val:.2f} 填充")
        
        # 处理分类变量的缺失值 - 用众数填充
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].isna().any():
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
                logger.info(f"列 '{col}' 用众数 '{mode_val}' 填充")
        
        self.cleaning_report['missing_values_handled'] = missing_info.to_dict()
        
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        转换数据类型
        Args:df: 输入数据框
        Returns:pd.DataFrame: 转换类型后的数据框
        """
        logger.info("转换数据类型")
        
        type_conversions = {}
        
        # 定义分类变量
        categorical_columns = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod', 'Churn'
        ]
        
        # 转换分类变量
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
                type_conversions[col] = 'category'
                logger.debug(f"列 '{col}' 转换为 category 类型")
        
        # 确保数值变量的类型
        numerical_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numerical_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                type_conversions[col] = 'numeric'
                logger.debug(f"列 '{col}' 转换为数值类型")
        
        self.cleaning_report['type_conversions'] = type_conversions
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        移除重复行
        Args:df: 输入数据框
        Returns:pd.DataFrame: 去重后的数据框
        """
        logger.info("检查并移除重复行")
        
        original_rows = len(df)
        df = df.drop_duplicates()
        removed_count = original_rows - len(df)
        
        if removed_count > 0:
            logger.info(f"移除了 {removed_count} 个重复行")
        else:
            logger.info("未发现重复行")
        
        self.cleaning_report['duplicates_removed'] = removed_count
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理异常值（根据配置决定是否处理）
        Args:df: 输入数据框
        Returns:pd.DataFrame: 处理异常值后的数据框
        """
        if not self.config.get('handle_outliers', False):
            logger.info("跳过异常值处理（配置中未启用）")
            return df
            
        logger.info("处理异常值")
        
        # 这里可以实现异常值检测和处理逻辑
        # 例如使用IQR方法检测和处理异常值
        
        outlier_report = {}
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                logger.info(f"列 '{col}' 发现 {outlier_count} 个异常值")
                outlier_report[col] = outlier_count
        
        self.cleaning_report['outliers_detected'] = outlier_report
        
        return df
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """
        获取数据清洗报告
        
        Returns:
            Dict[str, Any]: 清洗报告字典
        """
        return self.cleaning_report
    
    def print_cleaning_summary(self):
        """打印清洗摘要"""
        print("=" * 50)
        print("数据清洗摘要")
        print("=" * 50)
        print(f"原始数据形状: {self.cleaning_report.get('original_shape', 'N/A')}")
        print(f"清洗后数据形状: {self.cleaning_report.get('cleaned_shape', 'N/A')}")
        
        if 'duplicates_removed' in self.cleaning_report:
            print(f"移除重复行: {self.cleaning_report['duplicates_removed']}")
        
        if 'total_charges_converted' in self.cleaning_report:
            print(f"TotalCharges转换: {self.cleaning_report['total_charges_converted']}个值处理")


# 为了方便使用，也提供函数式接口
def clean_dataframe(df: pd.DataFrame, config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    函数式接口：清洗数据框
    
    Args:
        df: 原始数据框
        config: 配置参数
        
    Returns:
        Tuple[pd.DataFrame, Dict]: 清洗后的数据框和清洗报告
    """
    cleaner = DataCleaner(config)
    cleaned_df = cleaner.clean_data(df)
    report = cleaner.get_cleaning_report()
    
    return cleaned_df, report