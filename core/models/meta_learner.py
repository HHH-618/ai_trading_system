# core/models/meta_learner.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Dict, Tuple
from core.utils.memory import ReplayBuffer
from configs.settings import Config

class MetaLearner:
    """元学习框架，支持快速适应新市场条件"""
    
    def __init__(self, input_shape: Tuple[int, int]):
        self.input_shape = input_shape
        self.model = self._build_model()
        self.memory = ReplayBuffer(capacity=10000)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
    def _build_model(self) -> tf.keras.Model:
        """构建元学习模型架构"""
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # 共享特征提取层
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling1D(2)(x)
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.LSTM(64)(x)
        
        # 元学习分支
        meta_out = layers.Dense(64, activation='swish')(x)
        meta_out = layers.Dense(3, activation='softmax', name='meta_output')(meta_out)
        
        # 基础预测分支
        pred_out = layers.Dense(64, activation='swish')(x)
        pred_out = layers.Dense(1, activation='linear', name='pred_output')(pred_out)
        
        return models.Model(inputs=inputs, outputs=[meta_out, pred_out])
    
    def adapt(self, x: np.ndarray, y: np.ndarray, adaptation_steps: int = 3) -> None:
        """快速适应新数据"""
        for _ in range(adaptation_steps):
            with tf.GradientTape() as tape:
                pred = self.model(x)
                loss = self._compute_loss(y, pred)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: Tuple[np.ndarray, np.ndarray]) -> tf.Tensor:
        """计算复合损失函数"""
        meta_loss = tf.keras.losses.CategoricalCrossentropy()(y_true[0], y_pred[0])
        pred_loss = tf.keras.losses.Huber()(y_true[1], y_pred[1])
        return meta_loss + 0.5 * pred_loss
    
    def save(self, path: str = None) -> None:
        """保存模型"""
        if path is None:
            path = Config.MODEL_DIR / 'meta_learner.h5'
        self.model.save(path)
    
    def load(self, path: str = None) -> None:
        """加载模型"""
        if path is None:
            path = Config.MODEL_DIR / 'meta_learner.h5'
        self.model = tf.keras.models.load_model(path)
