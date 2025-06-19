# core/models/trainer.py

import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, List, Dict
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ReduceLROnPlateau,
    ModelCheckpoint
)
from configs.settings import Config
from configs.hyperparams import META_LEARNER_PARAMS, DQN_PARAMS
from utils.logger import setup_logger

logger = setup_logger('model_trainer')

class AdaptiveTrainer:
    """
    自适应模型训练系统，包含以下功能：
    1. 动态学习率调整
    2. 早停机制
    3. 梯度裁剪
    4. 混合精度训练
    5. 课程学习支持
    """
    
    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.best_weights = None
        self.current_lr = None
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: Dict[str, np.ndarray],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[Dict[str, np.ndarray]] = None,
        epochs: int = 100,
        batch_size: int = 64,
        initial_lr: float = 0.001,
        use_amp: bool = True,
        patience: int = 10
    ) -> tf.keras.callbacks.History:
        """执行自适应训练过程"""
        # 配置优化器
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self._build_lr_schedule(initial_lr),
            clipnorm=1.0  # 梯度裁剪
        )
        
        # 混合精度配置
        if use_amp:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        # 回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_direction_loss',
                patience=patience,
                restore_best_weights=True,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor='val_direction_loss',
                factor=0.5,
                patience=patience//2,
                mode='min'
            ),
            ModelCheckpoint(
                filepath=Config.MODEL_DIR / 'best_model.keras',
                save_best_only=True,
                monitor='val_direction_loss',
                mode='min'
            )
        ]
        
        # 训练模型
        history = self.model.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=self._calculate_batch_size(batch_size, X_train.shape[0]),
            callbacks=callbacks,
            verbose=2
        )
        
        self.best_weights = self.model.get_weights()
        return history
    
    def _build_lr_schedule(self, initial_lr: float) -> tf.keras.optimizers.schedules.LearningRateSchedule:
        """构建动态学习率计划"""
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )
    
    def _get_loss_function(self):
        """获取适合交易任务的损失函数"""
        return tf.keras.losses.Huber(
            delta=1.5,  # 对异常值更鲁棒
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )
    
    def _calculate_batch_size(self, base_size: int, dataset_size: int) -> int:
        """动态调整batch大小"""
        if dataset_size < 1000:
            return min(16, base_size)
        elif dataset_size < 5000:
            return min(32, base_size)
        return base_size
    
    def train_with_curriculum(
        self,
        X: np.ndarray,
        y: np.ndarray,
        difficulty_levels: int = 3,
        epochs_per_level: int = 20
    ) -> tf.keras.callbacks.History:
        """
        课程学习训练方法
        :param difficulty_levels: 难度级别数量
        :param epochs_per_level: 每个级别的训练轮数
        """
        histories = []
        
        for level in range(1, difficulty_levels + 1):
            logger.info(f"开始训练难度级别 {level}/{difficulty_levels}")
            
            # 根据难度级别筛选数据
            X_level, y_level = self._filter_by_difficulty(X, y, level, difficulty_levels)
            
            # 训练当前级别
            history = self.train(
                X_level, y_level,
                epochs=epochs_per_level,
                initial_lr=0.001 * (0.8 ** (level - 1))  # 逐步降低学习率
            )
            histories.append(history)
            
        return histories
    
    def _filter_by_difficulty(self, X: np.ndarray, y: np.ndarray, 
                            current_level: int, total_levels: int) -> Tuple[np.ndarray, np.ndarray]:
        """根据难度级别筛选数据"""
        # 简化的难度划分 - 实际应用中可以根据波动率、趋势等指标划分
        chunk_size = len(X) // total_levels
        start = (current_level - 1) * chunk_size
        end = current_level * chunk_size if current_level < total_levels else len(X)
        return X[start:end], y[start:end]
    
    def evaluate_model(self, X: np.ndarray, y: np.ndarray, 
                      metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """评估模型性能"""
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'auc']
        
        results = {}
        for metric in metrics:
            if metric == 'accuracy':
                _, acc = self.model.evaluate(X, y, verbose=0)
                results['accuracy'] = acc
            # 可以添加其他自定义指标计算
        
        return results
