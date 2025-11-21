# config.py
import yaml
from pathlib import Path
from typing import Dict, Any

def get_config(config_path: str = None) -> Dict[str, Any]:
    """
    获取配置参数
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Dict[str, Any]: 配置字典
    """
    # 默认配置
    default_config = {
        'data_path': 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv',
        'save_cleaned_data': True,
        'generate_plots': True,
        'random_seed': 42,
        
        'data_cleaning': {
            'handle_outliers': False,
            'remove_duplicates': True
        },
        
        'eda': {
            'correlation_threshold': 0.5,
            'top_categories_limit': 10
        },
        
        'modeling': {
            'test_size': 0.2,
            'cv_folds': 5
        }
    }
    
    # 如果提供了配置文件路径，则从文件加载配置
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as file:
                file_config = yaml.safe_load(file)
            
            # 合并配置（文件配置覆盖默认配置）
            if file_config:
                default_config.update(file_config)
                
        except Exception as e:
            print(f"加载配置文件失败: {e}，使用默认配置")
    
    return default_config

# 如果使用YAML配置文件，可以创建 config.yaml 文件
def save_default_config(config_path: str = 'config.yaml'):
    """
    保存默认配置到YAML文件
    
    Args:
        config_path: 配置文件保存路径
    """
    default_config = get_config()
    
    with open(config_path, 'w') as file:
        yaml.dump(default_config, file, default_flow_style=False)
    
    print(f"默认配置已保存到: {config_path}")


if __name__ == "__main__":
    # 如果需要生成默认配置文件，运行此脚本
    save_default_config()