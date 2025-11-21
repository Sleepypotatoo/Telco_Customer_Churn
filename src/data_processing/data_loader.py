# src/data_processing/data_loader.py
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataLoader:
    """
    数据加载类
    """
    
    def __init__(self):
        pass
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        加载数据文件
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            pd.DataFrame: 加载的数据框
            
        Raises:
            FileNotFoundError: 当文件不存在时
            Exception: 其他加载错误
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"数据文件不存在: {file_path}")
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        logger.info(f"正在加载数据文件: {file_path}")
        
        try:
            # 根据文件扩展名选择加载方法
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                # 尝试用CSV格式读取
                df = pd.read_csv(file_path)
                
            logger.info(f"成功加载数据，形状: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"加载数据文件失败: {e}")
            raise