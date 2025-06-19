# core/strategies/supervised/lstm.py

import numpy as np
import tensorflow as tf
from typing import Optional, Tuple
import pandas as pd
from core.strategies.base import BaseStrategy
from configs.constants import ModelConstants
from utils.logger import setup_logger

logger = setup_logger('lstm_strategy')

class LSTMStrategy(BaseStrategy):
    """
    基于LSTM的监督学习策略
    使用多层LSTM网络预测价格走势
    """
    
    def __init__(self, model_path: str = None):
        super().__init__()
        self.required_indicators = ['close', 'EMA10', 'RSI', 'MACD']
        self.min_data_length = ModelConstants.SEQ_LENGTH * 2
        self.model = self._load_model(model_path) if model_path else None
        self.scaler = None  # 应在训练后设置
        
    def generate_signal(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
        """生成LSTM预测信号"""
        if not self._validate_data(df):
            return None, None
            
        if self.model is None:
            logger.warning("LSTM模型未加载")
            return None, None
            
        try:
            # 1. 预处理数据
            processed_data = self._preprocess_data(df)
            if processed_data is None:
                return None, None
                
            # 2. 模型预测
            prediction = self.model.predict(processed_data[np.newaxis, ...], verbose=0)
            direction_prob, magnitude = prediction[0][0], prediction[1][0]
            
            # 3. 生成信号
            if direction_prob > 0.7:  # 买入信号
                return 'buy', float(direction_prob)
            elif direction_prob < 0.3:  # 卖出信号
                return 'sell', float(1 - direction_prob)
            else:
                return None, None
                
        except Exception as e:
            logger.error(f"LSTM预测异常: {str(e)}")
            return None, None
            
    def _preprocess_data(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """增强版数据预处理"""
        try:
            # 1. 检查并填充缺失值
            df = df[self.required_indicators].ffill().bfill()
        
            # 2. 计算额外特征
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['log_return'].rolling(5).std()
        
            # 3. 选择最终特征
            features = df[[
                'close',
                'EMA10',
                'EMA20',
                'RSI',
                'MACD',
                'log_return',
                'volatility'
            ]].values[-ModelConstants.SEQ_LENGTH:]
        
            # 4. 标准化处理
            if self.scaler is None:
                from sklearn.preprocessing import RobustScaler
                self.scaler = RobustScaler()
                self.scaler.fit(features)
        
            # 5. 处理序列长度
            if len(features) < ModelConstants.SEQ_LENGTH:
                pad_len = ModelConstants.SEQ_LENGTH - len(features)
                features = np.pad(features, ((pad_len, 0), (0, 0)), 'edge')
        
            return self.scaler.transform(features)
        
        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")
            return None
            
    def _load_model(self, model_path: str) -> Optional[tf.keras.Model]:
        """加载预训练LSTM模型"""
        try:
            model = tf.keras.models.load_model(model_path)
            logger.info(f"成功加载LSTM模型: {model_path}")
            return model
        except Exception as e:
            logger.error(f"加载LSTM模型失败: {str(e)}")
            return None
            
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """构建LSTM模型架构"""
        inputs = tf.keras.Input(shape=input_shape)
        
        # 双向LSTM层
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)
        )(inputs)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32)
        )(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # 多任务输出
        direction_out = tf.keras.layers.Dense(1, activation='sigmoid', name='direction')(x)
        magnitude_out = tf.keras.layers.Dense(1, activation='linear', name='magnitude')(x)
        
        model = tf.keras.Model(
            inputs=inputs,
            outputs=[direction_out, magnitude_out]
        )

        metrics = {
            'direction': ['accuracy', tf.keras.metrics.AUC(name='auc')],
            'magnitude': ['mae', 'mse']
        }
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'direction': 'binary_crossentropy',
                'magnitude': tf.keras.losses.Huber()
            },
            metrics=metrics,
            loss_weights={
                'direction': 0.7,  # 方向预测权重
                'magnitude': 0.3    # 幅度预测权重
            }
        )
        
        return model

    def save_model(self, path: str):
        """保存模型和标准化器"""
        import joblib
    
        # 1. 创建目录
        import os
        os.makedirs(path, exist_ok=True)
    
        # 2. 保存模型
        model_path = os.path.join(path, 'lstm_model.keras')
        self.model.save(model_path)
    
        # 3. 保存标准化器
        if self.scaler is not None:
            scaler_path = os.path.join(path, 'scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
    
        logger.info(f"模型已保存到 {path}")

    @classmethod
    def load_model(cls, path: str):
        """加载模型和标准化器"""
        import joblib
        import os
    
        # 1. 加载模型
        model_path = os.path.join(path, 'lstm_model.keras')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
        strategy = cls()
        strategy.model = tf.keras.models.load_model(model_path)
    
        # 2. 加载标准化器
        scaler_path = os.path.join(path, 'scaler.pkl')
        if os.path.exists(scaler_path):
            strategy.scaler = joblib.load(scaler_path)
    
        logger.info(f"模型从 {path} 加载成功")
        return strategy
