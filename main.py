#!/usr/bin/env python3
"""
电信客户流失分析主脚本
功能：加载 → 清洗 → 可视化 → 基础特征 → 高级特征 → 特征选择 → 保存结果
"""
import sys
from pathlib import Path
import logging
import pandas as pd
# ---------- 导入纯函数（不再导入类） ----------
from src.data_processing.data_cleaner import clean_data
from src.data_processing.eda import EDA
from src.feature_engineering.basic_features import create_basic_features
from src.feature_engineering.advanced_features import create_advanced_features
from src.feature_engineering.feature_selection import select_correlation
from src.reporting.quality_report import quality_report
from src.reporting.numerical_report import numerical_report
from src.reporting.feature_documentation import generate_feature_documentation, save_feature_info_json

# ---------- 路径加入 ----------
sys.path.append(str(Path(__file__).parent / 'src'))

# ---------- 日志配置（统一用标准 logging） ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('telco_churn_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)


# ---------- 目录初始化 ----------
def init_dirs():
    dirs = ['reports/plots', 'data', 'models']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
    logging.info("[初始化] 必要目录已确认")


# ---------- 1. 加载数据 ----------
def load_data() -> pd.DataFrame:
    csv_path = Path('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    if not csv_path.exists():
        raise FileNotFoundError(f"请把原始数据放到 {csv_path}")
    df = pd.read_csv(csv_path)
    logging.info(f"[加载] 数据形状：{df.shape}")
    return df


# ---------- 2. 数据清洗 ----------
def run_clean(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = clean_data(df)  # 直接调纯函数
    out_path = Path('data/cleaned.csv')
    df_clean.to_csv(out_path, index=False)
    logging.info(f"[清洗] 已保存清洗结果 -> {out_path}")
    return df_clean


# ---------- 3. 可视化 ----------
def run_eda(df: pd.DataFrame):
    eda = EDA()  # 仅保留 EDA 类当调度器，内部仍调纯函数画图
    paths = eda.perform_visual_analysis(df)
    logging.info(f"[可视化] 共生成 {len(paths)} 张图 -> reports/plots/")


# ---------- 4. 特征工程 ----------
def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # 4.1 基础特征
    df_base = create_basic_features(df)
    logging.info(f"[特征] 基础特征完成，列数：{df_base.shape[1]}")

    # 4.2 高级特征
    df_adv = create_advanced_features(df_base)
    logging.info(f"[特征] 高级特征完成，列数：{df_adv.shape[1]}")

    # 4.3 特征选择（相关性过滤）
    if 'Churn_numeric' in df_adv.columns:
        df_selected = select_correlation(df_adv, target_col='Churn_numeric', threshold=0.05)
        logging.info(f"[特征] 相关性选择完成，列数：{df_selected.shape[1]}")
        # 保存
        out_path = Path('data/engineered.csv')
        df_selected.to_csv(out_path, index=False)
        logging.info(f"[特征] 已保存特征工程结果 -> {out_path}")
        return df_selected

    return df_adv

def generate_feature_documentation_report(engineered_data: pd.DataFrame):
    """生成特征文档"""
    logging.info("生成特征文档说明")
    
    try:
        # 生成Markdown格式的特征文档
        generate_feature_documentation(engineered_data, "reports/feature_documentation.md")
        
        # 生成JSON格式的特征信息
        from src.reporting.feature_documentation import analyze_features, save_feature_info_json
        feature_info = analyze_features(engineered_data)
        save_feature_info_json(feature_info, "reports/feature_info.json")
        
        logging.info("特征文档生成完成")
        
    except Exception as e:
        logging.error(f"生成特征文档失败: {e}")


# ---------- 主流程 ----------
def main():
    init_dirs()
    logging.info("=" * 60)
    logging.info("电信客户流失分析开始")
    logging.info("=" * 60)

    # 1. 加载
    df_raw = load_data()
    # 2. 清洗
    df_clean = run_clean(df_raw)
    # 3. 可视化
    run_eda(df_clean)
    # 4. 特征工程
    df_engineered = run_feature_engineering(df_clean)
    # 5. 生成特征文档
    generate_feature_documentation_report(df_engineered)

    logging.info("=" * 60)
    logging.info("全部完成！查看：")
    logging.info("- 日志：telco_churn_analysis.log")
    logging.info("- 图表：reports/plots/")
    logging.info("- 清洗数据：data/cleaned.csv")
    logging.info("- 特征数据：data/engineered.csv")
    logging.info("- 特征文档：reports/feature_documentation.md")
    logging.info("=" * 60)

    quality_report(df_clean)          # 清洗后数据
    numerical_report(df_clean) 
    


if __name__ == '__main__':
    main()
    