# core/strategies/supervised/transformer.py

import numpy as np
import tensorflow as tf
from typing import Optional, Tuple
import pandas as pd
from core.strategies.base import BaseStrategy
from configs.constants import ModelConstants
from utils.logger import setup_logger

logger = setup_logger('transformer_strategy')

class TransformerStrategy(BaseStrategy):
    """
    基于Transformer的时间序列预测策略
    使用注意力机制捕捉长期依赖关系
    """
    
    def __init__(self, model_path: str = None):
        super().__init__()
        self.required_indicators = ['close', 'EMA10', 'EMA20', 'RSI', 'MACD', 'ATR']
        self.min_data_length = ModelConstants.SEQ_LENGTH * 2
        self.model = self._load_model(model_path) if model_path else None
        self.scaler = None  # 应在训练后设置
        
    def generate_signal(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
        """生成Transformer预测信号"""
        if not self._validate_data(df):
            return None, None
            
        if self.model is None:
            logger.warning("Transformer模型未加载")
            return None, None
            
        try:
            # 1. 预处理数据
            processed_data = self._preprocess_data(df)
            if processed_data is None:
                return None, None
                
            # 2. 模型预测
            prediction = self.model.predict(processed_data[np.newaxis, ...], verbose=0)
            direction_prob, magnitude = prediction[0][0], prediction[1][0]
            
            # 3. 动态置信度调整
            confidence = self._adjust_confidence(direction_prob, volatility)
            
            # 4. 生成信号
            if direction_prob > 0.65 and confidence > 0.6:  # 买入信号
                return 'buy', confidence
            elif direction_prob < 0.35 and confidence > 0.6:  # 卖出信号
                return 'sell', confidence
            else:
                return None, None
                
        except Exception as e:
            logger.error(f"Transformer预测异常: {str(e)}")
            return None, None
            
    def _adjust_confidence(self, direction_prob: float, volatility: float) -> float:
        """基于波动率调整置信度"""
        base_confidence = max(direction_prob, 1 - direction_prob)
        volatility_factor = 1 / (1 + np.exp(volatility - 1))  # sigmoid调整
        return float(base_confidence * volatility_factor)
        
    def _preprocess_data(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """准备Transformer输入数据"""
        try:
            # 选择特征列
            features = df[self.required_indicators].values[-ModelConstants.SEQ_LENGTH:]
            
            # 标准化
            if self.scaler is not None:
                features = self.scaler.transform(features)
                
            return features.astype(np.float32)
            
        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")
            return None
            
    def _load_model(self, model_path: str) -> Optional[tf.keras.Model]:
        """加载预训练Transformer模型"""
        try:
            model = tf.keras.models.load_model(model_path)
            logger.info(f"成功加载Transformer模型: {model_path}")
            return model
        except Exception as e:
            logger.error(f"加载Transformer模型失败: {str(e)}")
            return None
            
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """构建Transformer模型架构"""
        inputs = tf.keras.Input(shape=input_shape)
        
        # 位置编码
        positions = tf.range(start=0, limit=input_shape[0], delta=1)
        position_embedding = tf.keras.layers.Embedding(
            input_dim=input_shape[0],
            output_dim=input_shape[1]
        )(positions)
        
        # 添加位置信息
        x = inputs + position_embedding
        
        # Transformer编码器层
        x = tf.keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=input_shape[1]
        )(x, x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        
        # 前馈网络
        x = tf.keras.layers.Dense(64, activation='gelu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        
        # 全局平均池化
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # 多任务输出
        direction_out = tf.keras.layers.Dense(1, activation='sigmoid', name='direction')(x)
        magnitude_out = tf.keras.layers.Dense(1, activation='linear', name='magnitude')(x)
        
        model = tf.keras.Model(
            inputs=inputs,
            outputs=[direction_out, magnitude_out]
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss={
                'direction': 'binary_crossentropy',
                'magnitude': tf.keras.losses.Huber()
            },
            metrics={
                'direction': ['accuracy', tf.keras.metrics.AUC()],
                'magnitude': ['mae']
            }
        )
        
        return model
