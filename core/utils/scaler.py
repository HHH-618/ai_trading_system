# core/utils/scaler.py

import numpy as np
import joblib
from typing import Optional, Dict
from sklearn.preprocessing import (
    RobustScaler,
    MinMaxScaler,
    StandardScaler,
    QuantileTransformer
)
from utils.logger import setup_logger

logger = setup_logger('data_scaler')

class AdaptiveScaler:
    """
    自适应数据标准化器，支持:
    1. 动态特征分组标准化
    2. 自动处理离群值
    3. 多种标准化方法
    4. 增量学习
    """
    
    def __init__(self, method: str = 'robust'):
        """
        :param method: 标准化方法 (robust/minmax/zscore/quantile)
        """
        self.method = method
        self.scalers = {}  # 按特征组存储scaler
        self.feature_groups = {}  # 特征分组配置
        
    def fit(self, X: np.ndarray, feature_groups: Dict[str, list]):
        """
        拟合标准化器
        :param X: 输入数据
        :param feature_groups: 特征分组 {'group_name': [col_indices]}
        """
        self.feature_groups = feature_groups
        
        for group_name, cols in feature_groups.items():
            if self.method == 'robust':
                scaler = RobustScaler()
            elif self.method == 'minmax':
                scaler = MinMaxScaler()
            elif self.method == 'zscore':
                scaler = StandardScaler()
            elif self.method == 'quantile':
                scaler = QuantileTransformer(output_distribution='normal')
            else:
                raise ValueError(f"未知的标准化方法: {self.method}")
                
            # 只拟合指定列
            scaler.fit(X[:, cols])
            self.scalers[group_name] = scaler
            
    def transform(self, X: np.ndarray) -> np.ndarray:
        """应用标准化转换"""
        X_transformed = X.copy()
        
        for group_name, cols in self.feature_groups.items():
            if group_name in self.scalers:
                X_transformed[:, cols] = self.scalers[group_name].transform(X[:, cols])
                
        return X_transformed
        
    def partial_fit(self, X: np.ndarray):
        """增量学习"""
        for group_name, cols in self.feature_groups.items():
            if group_name in self.scalers:
                self.scalers[group_name].partial_fit(X[:, cols])
                
    def save(self, filepath: str):
        """保存scaler到文件"""
        joblib.dump({
            'method': self.method,
            'scalers': self.scalers,
            'feature_groups': self.feature_groups
        }, filepath)
        
    @classmethod
    def load(cls, filepath: str) -> 'AdaptiveScaler':
        """从文件加载scaler"""
        data = joblib.load(filepath)
        scaler = cls(data['method'])
        scaler.scalers = data['scalers']
        scaler.feature_groups = data['feature_groups']
        return scaler
        
    def get_feature_ranges(self) -> Dict[str, tuple]:
        """获取各特征组的数值范围"""
        ranges = {}
        
        for group_name, scaler in self.scalers.items():
            if hasattr(scaler, 'scale_'):
                # RobustScaler/StandardScaler
                center = scaler.center_ if hasattr(scaler, 'center_') else 0
                scale = scaler.scale_
                ranges[group_name] = (
                    center - 3*scale,  # 近似下限
                    center + 3*scale   # 近似上限
                )
            elif hasattr(scaler, 'data_min_'):
                # MinMaxScaler
                ranges[group_name] = (scaler.data_min_, scaler.data_max_)
                
        return ranges
