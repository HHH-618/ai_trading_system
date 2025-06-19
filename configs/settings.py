# configs/settings.py

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()
class ModelConstants:
    SEQ_LENGTH = int(os.getenv('SEQ_LENGTH', 60))   # 序列长度
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 64))   # 训练批次大小
    VALIDATION_SPLIT = 0.2        # 验证集比例

class Config:
    # 基础配置
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    MODEL_DIR = BASE_DIR / 'models'
    LOG_DIR = BASE_DIR / 'logs'
    
    # 动态配置加载
    @classmethod
    def load_config(cls) -> Dict[str, Any]:
        return {
            'trading': {
                'symbols': os.getenv('SYMBOLS', 'XAUUSD,EURUSD').split(','),
                'timeframes': ['M5', 'H1', 'D1'],
                'max_drawdown': float(os.getenv('MAX_DRAWDOWN', '0.2')),
                'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '0.01'))
            },
            'model': {
                'train_interval': os.getenv('TRAIN_INTERVAL', '1d'),
                'retrain_threshold': float(os.getenv('RETRAIN_THRESHOLD', '0.7'))
            }
        }
    
    # 自动创建目录
    @classmethod
    def init_dirs(cls):
        for d in [cls.DATA_DIR, cls.MODEL_DIR, cls.LOG_DIR]:
            d.mkdir(exist_ok=True)

Config.init_dirs()
