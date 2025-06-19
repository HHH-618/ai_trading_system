# core/models/ensemble.py

import numpy as np
import tensorflow as tf
from typing import List, Dict, Optional
from configs.hyperparams import META_LEARNER_PARAMS
from utils.logger import setup_logger

logger = setup_logger('model_ensemble')

class ModelEnsemble:
    """
    高级模型集成系统，支持动态权重调整和模型选择
    实现以下集成方法：
    1. 加权平均
    2. 堆叠泛化
    3. 动态选择
    """
    
    def __init__(self, models: List[tf.keras.Model]):
        self.models = models
        self.weights = np.ones(len(models)) / len(models)  # 初始等权重
        self.performance_history = []
        self.adjustment_factor = 0.1  # 权重调整步长
        
    def weighted_predict(self, X: np.ndarray) -> np.ndarray:
        """加权平均预测"""
        predictions = np.zeros((X.shape[0], 1))
        for i, model in enumerate(self.models):
            pred = model.predict(X, verbose=0)
            predictions += self.weights[i] * pred
        return predictions
    
    def stacked_predict(self, X: np.ndarray, meta_model: tf.keras.Model) -> np.ndarray:
        """堆叠泛化预测"""
        meta_features = np.column_stack([
            model.predict(X, verbose=0) for model in self.models
        ])
        return meta_model.predict(meta_features, verbose=0)
    
    def dynamic_select_predict(self, X: np.ndarray, 
                             market_conditions: Dict[str, float]) -> np.ndarray:
        """
        基于市场条件动态选择模型
        :param market_conditions: 包含波动率、趋势强度等指标
        """
        # 根据市场状态计算模型得分
        scores = self._calculate_model_scores(market_conditions)
        best_model_idx = np.argmax(scores)
        return self.models[best_model_idx].predict(X, verbose=0)
    
    def _calculate_model_scores(self, market_conditions: Dict[str, float]) -> np.ndarray:
        """计算各模型在当前市场条件下的得分"""
        scores = np.zeros(len(self.models))
        
        # 模型1在低波动市场表现更好
        scores[0] = 1 - market_conditions['volatility']
        
        # 模型2在强趋势市场表现更好
        scores[1] = market_conditions['trend_strength']
        
        # 模型3在高波动市场表现更好
        scores[2] = market_conditions['volatility']
        
        return scores
    
    def update_weights(self, recent_performance: List[float]):
        """
        根据近期表现动态调整模型权重
        :param recent_performance: 各模型近期表现指标(如夏普比率)
        """
        self.performance_history.append(recent_performance)
        
        # 指数加权移动平均
        if len(self.performance_history) > 1:
            performance_trend = np.diff(self.performance_history, axis=0).mean(axis=0)
            adjustment = performance_trend * self.adjustment_factor
            new_weights = self.weights + adjustment
            self.weights = np.exp(new_weights) / np.sum(np.exp(new_weights))  # softmax
            
        logger.info(f"更新模型权重: {self.weights}")
    
    def save_ensemble(self, path: str):
        """保存集成模型"""
        for i, model in enumerate(self.models):
            model.save(f"{path}/model_{i}.h5")
        np.save(f"{path}/weights.npy", self.weights)
    
    @classmethod
    def load_ensemble(cls, path: str):
        """加载集成模型"""
        import glob
        model_files = glob.glob(f"{path}/model_*.h5")
        models = [tf.keras.models.load_model(f) for f in sorted(model_files)]
        weights = np.load(f"{path}/weights.npy")
        ensemble = cls(models)
        ensemble.weights = weights
        return ensemble
