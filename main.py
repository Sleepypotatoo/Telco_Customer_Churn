# main.py
import os
import sys
import logging
import pandas as pd
from pathlib import Path

# 添加src目录到Python路径，以便可以导入模块
sys.path.append(str(Path(__file__).parent / "src"))

from config import get_config
from src.data_processing.data_loader import DataLoader
from src.data_processing.data_cleaner import DataCleaner
from src.data_processing.eda import EDA
#from src.visualization.eda_plots import EDAPlots

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('telco_churn_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TelcoChurnAnalysis:
    """
    电信客户流失分析主类
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化分析流程
        
        Args:
            config_path: 配置文件路径
        """
        self.config = get_config(config_path)
        self.data_loader = DataLoader()
        self.data_cleaner = DataCleaner(self.config.get('data_cleaning', {}))
        self.eda = EDA()
        
        # 创建必要的目录
        self._create_directories()
        
    def _create_directories(self):
        """创建必要的目录结构"""
        directories = ['models', 'reports', 'reports/plots', 'reports/tables']
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"确保目录存在: {directory}")
    
    def run_full_analysis(self):
        """
        运行完整的分析流程
        """
        logger.info("开始电信客户流失分析流程")
        
        try:
            # 1. 数据加载
            raw_data = self.load_data()
            
            # 2. 数据清洗
            cleaned_data = self.clean_data(raw_data)
            
            # 3. 探索性数据分析
            self.perform_eda(cleaned_data)
            
            # 4. 特征工程（后续添加）
            # engineered_data = self.perform_feature_engineering(cleaned_data)
            
            # 5. 建模（后续添加）
            # self.perform_modeling(engineered_data)
            
            logger.info("分析流程完成")
            
        except Exception as e:
            logger.error(f"分析流程出错: {e}")
            raise
    
    def load_data(self) -> pd.DataFrame:
        """
        加载数据
        
        Returns:
            pd.DataFrame: 原始数据
        """
        logger.info("步骤1: 数据加载")
        
        data_path = self.config['data_path']
        logger.info(f"从 {data_path} 加载数据")
        
        # 使用DataLoader加载数据
        raw_data = self.data_loader.load_data(data_path)
        
        # 显示数据基本信息
        logger.info(f"数据形状: {raw_data.shape}")
        logger.info(f"数据列: {list(raw_data.columns)}")
        logger.info(f"数据前5行:\n{raw_data.head()}")
        
        return raw_data
    
    def clean_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        清洗数据   
        Args:raw_data: 原始数据    
        Returns:pd.DataFrame: 清洗后的数据
        """
        logger.info("步骤2: 数据清洗")
        
        # 使用DataCleaner清洗数据
        cleaned_data = self.data_cleaner.clean_data(raw_data)
        
        # 显示清洗摘要
        self.data_cleaner.print_cleaning_summary()
        
        # 保存清洗后的数据
        if self.config.get('save_cleaned_data', True):
            cleaned_data_path = 'data/cleaned_telco_churn.csv'
            cleaned_data.to_csv(cleaned_data_path, index=False)
            logger.info(f"清洗后的数据已保存到: {cleaned_data_path}")
        
        return cleaned_data
    
    def perform_eda(self, data: pd.DataFrame):
        """
        执行探索性数据分析
        Args:data: 清洗后的数据
        """
        logger.info("步骤3: 探索性数据分析")
        
        # 执行基本EDA
        eda_report = self.eda.perform_eda(data)
        
        # 生成可视化图表
        if self.config.get('generate_plots', True):
            self.generate_visualizations(data, eda_report)
        
        # 保存EDA报告
        self.save_eda_report(eda_report)
    
    # def generate_visualizations(self, data: pd.DataFrame, eda_report: dict):
    #     """
    #     生成可视化图表
    #     Args:data: 数据
    #     eda_report: EDA报告
    #     """
    #     logger.info("生成可视化图表")
        
    #     try:
    #         plotter = EDAPlots()
            
    #         # 生成各种图表
    #         plotter.plot_target_distribution(data, save_path='reports/plots/target_distribution.png')
    #         plotter.plot_numerical_distributions(data, save_path='reports/plots/numerical_distributions.png')
    #         plotter.plot_categorical_distributions(data, save_path='reports/plots/categorical_distributions.png')
    #         plotter.plot_correlation_heatmap(data, save_path='reports/plots/correlation_heatmap.png')
            
    #         logger.info("可视化图表已生成并保存到 reports/plots/ 目录")
            
    #     except Exception as e:
    #         logger.warning(f"生成可视化图表时出错: {e}")
    
    def save_eda_report(self, eda_report: dict):
        """
        保存EDA报告
        Args:eda_report: EDA报告字典
        """
        logger.info("保存EDA报告")
        
        try:
            # 保存基本统计信息
            if 'basic_stats' in eda_report:
                basic_stats_df = pd.DataFrame(eda_report['basic_stats'])
                basic_stats_df.to_csv('reports/tables/basic_statistics.csv', index=False)
            
            # 保存缺失值报告
            if 'missing_values' in eda_report:
                missing_df = pd.DataFrame(eda_report['missing_values'])
                missing_df.to_csv('reports/tables/missing_values_report.csv', index=False)
            
            # 保存其他报告...
            
            logger.info("EDA报告已保存到 reports/tables/ 目录")
            
        except Exception as e:
            logger.warning(f"保存EDA报告时出错: {e}")
    
    def perform_feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        执行特征工程（待实现）
        
        Args:
            data: 清洗后的数据
            
        Returns:
            pd.DataFrame: 特征工程后的数据
        """
        logger.info("步骤4: 特征工程")
        
        # 这里将调用特征工程模块
        # 暂时返回原始数据，后续实现
        logger.info("特征工程模块待实现")
        
        return data
    
    def perform_modeling(self, data: pd.DataFrame):
        """
        执行建模流程（待实现）
        
        Args:
            data: 特征工程后的数据
        """
        logger.info("步骤5: 建模")
        
        # 这里将调用建模模块
        logger.info("建模模块待实现")


def main():
    """
    主函数
    """
    print("=" * 60)
    print("电信客户流失预测分析系统")
    print("=" * 60)
    
    try:
        # 初始化分析系统
        analysis = TelcoChurnAnalysis()
        
        # 运行完整分析流程
        analysis.run_full_analysis()
        
        print("\n" + "=" * 60)
        print("分析完成！")
        print("请查看以下内容：")
        print("- 日志文件: telco_churn_analysis.log")
        print("- 可视化图表: reports/plots/")
        print("- 数据表格: reports/tables/")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"主程序运行出错: {e}")
        print(f"程序执行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()