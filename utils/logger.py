# utils/logger.py

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from configs.settings import Config

class ColorFormatter(logging.Formatter):
    """带颜色的日志格式化器"""
    COLORS = {
        'DEBUG': '\033[36m',     # 青色
        'INFO': '\033[32m',      # 绿色
        'WARNING': '\033[33m',   # 黄色
        'ERROR': '\033[31m',     # 红色
        'CRITICAL': '\033[1;31m' # 红色加粗
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        message = super().format(record)
        return f"{color}{message}{self.RESET}" if color else message

def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_to_file: bool = True,
    file_level: Optional[int] = None,
    propagate: bool = False
) -> logging.Logger:
    """
    配置并返回一个logger实例
    :param name: logger名称
    :param level: 控制台日志级别
    :param log_to_file: 是否写入文件
    :param file_level: 文件日志级别(默认与控制台相同)
    :param propagate: 是否传播到父logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = ColorFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 文件handler
    if log_to_file:
        file_level = file_level if file_level is not None else level
        log_dir = Config.LOG_DIR
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

def log_performance_metrics(metrics: dict, logger_name: str = 'performance'):
    """
    记录性能指标到专用日志文件
    :param metrics: 指标字典 {'metric_name': value}
    :param logger_name: 专用logger名称
    """
    perf_logger = logging.getLogger(logger_name)
    if not perf_logger.handlers:
        setup_logger(logger_name, log_to_file=True, file_level=logging.INFO)
    
    metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    perf_logger.info(metric_str)
