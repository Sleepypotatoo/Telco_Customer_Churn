# src/data_processing/eda.py - 简化版，只负责可视化
import logging
from src.visualization.eda_plots import EDAPlots

logger = logging.getLogger(__name__)

class EDA:
    """探索性数据分析类 - 只负责可视化"""
    
    def __init__(self):
        self.plotter = EDAPlots()
    
    def perform_visual_analysis(self, df):
        """
        执行可视化分析，生成所有图表
        Args:df: 清洗后的数据
        Returns:dict: 图表文件路径信息
        """
        logger.info("开始可视化分析")
        
        plot_paths = {}
        
        try:
            # 基础分布图表
            plot_paths['target_dist'] = self.plotter.plot_target_distribution(
                df, 'reports/plots/target_distribution.png')
            
            plot_paths['numerical_dist'] = self.plotter.plot_numerical_distributions(
                df, 'reports/plots/numerical_distributions.png')
            
            plot_paths['categorical_dist'] = self.plotter.plot_categorical_distributions(
                df, 'reports/plots/categorical_distributions.png')
            
            # 高级分析图表
            plot_paths['churn_rates'] = self.plotter.plot_churn_rates_by_features(
                df, 'reports/plots/churn_rates_by_features.png')
            
            plot_paths['tenure_vs_churn'] = self.plotter.plot_tenure_vs_churn(
                df, 'reports/plots/tenure_vs_churn.png')
            
            plot_paths['charges_vs_churn'] = self.plotter.plot_charges_vs_churn(
                df, 'reports/plots/charges_vs_churn.png')
            
            plot_paths['services_usage'] = self.plotter.plot_services_usage(
                df, 'reports/plots/services_usage.png')
            
            logger.info("可视化分析完成")
            return plot_paths
            
        except Exception as e:
            logger.error(f"可视化分析失败: {e}")
            return {}