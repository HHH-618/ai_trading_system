# core/strategies/rl/dqn.py

import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import deque
from typing import Dict, Any, Tuple, Optional, List
from core.utils.memory import PrioritizedReplayBuffer
from core.strategies.base import BaseStrategy
from utils.logger import setup_logger

logger = setup_logger('dqn_strategy')

class DQNStrategy(BaseStrategy):
    """基于深度Q学习的交易策略"""
    
    def __init__(self, state_shape: Tuple[int, int], action_size: int = 3):
        super().__init__()
        self.state_shape = state_shape
        self.action_size = action_size  # 0: hold, 1: buy, 2: sell
        self.memory = PrioritizedReplayBuffer(capacity=5000) 
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        
    def _build_model(self) -> tf.keras.Model:
        """构建DQN网络"""
        # 启用混合精度训练
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)

        # 构建模型
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, input_shape=self.state_shape, return_sequences=True),
            tf.keras.layers.LSTM(16),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear', dtype='float32')
        ])

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.00025,  # 降低学习率
            clipnorm=1.0  # 添加梯度裁剪
        )
        
        model.compile(optimizer=optimizer, loss='huber')
        return model
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_model.set_weights(self.model.get_weights())
        
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        experience = (state, action, reward, next_state, done)
        self.memory.add(experience)
        
    def act(self, state: np.ndarray) -> int:
        logger.info(f"Act函数调用 - 输入状态形状: {state.shape}")
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.action_size)
            logger.debug(f"随机选择动作: {action}")
            return action

        logger.info(f"模型预测前 - 重塑后状态形状: {state[np.newaxis, ...].shape}")
        act_values = self.model.predict(state[np.newaxis, ...], verbose=0)
        logger.info(f"模型预测完成 - 输出: {act_values}")
        return np.argmax(act_values[0])

    def save(self, path: str):
        """保存模型和必要组件"""
        import os
        from pathlib import Path

        logger.info(f"保存DQN模型到 {path}")
    
        # 确保目录存在
        Path(path).mkdir(parents=True, exist_ok=True)
    
        # 保存模型
        self.model.save(os.path.join(path, 'dqn_model.keras'))
    
        # 保存目标模型
        self.target_model.save(os.path.join(path, 'dqn_target_model.keras'))
    
        # 保存其他必要状态
        import pickle
        state = {
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'state_shape': self.state_shape,
            'action_size': self.action_size
        }
    
        with open(os.path.join(path, 'dqn_state.pkl'), 'wb') as f:
            pickle.dump(state, f)
    
        # 如果有scaler也保存
        if hasattr(self, 'scaler'):
            with open(os.path.join(path, 'dqn_scaler.pkl'), 'wb') as f:
                pickle.dump(self.scaler, f)

    @classmethod
    def load(cls, path: str):
        """加载保存的模型"""
        import os
        import pickle
    
        # 加载状态
        with open(os.path.join(path, 'dqn_state.pkl'), 'rb') as f:
            state = pickle.load(f)
    
        # 创建新实例
        instance = cls(state['state_shape'], state['action_size'])
    
        # 恢复状态
        instance.epsilon = state['epsilon']
        instance.gamma = state['gamma']
        instance.epsilon_decay = state['epsilon_decay']
        instance.epsilon_min = state['epsilon_min']
    
        # 加载模型
        instance.model = tf.keras.models.load_model(os.path.join(path, 'dqn_model.keras'))
        instance.target_model = tf.keras.models.load_model(os.path.join(path, 'dqn_target_model.keras'))
    
        # 加载scaler
        scaler_path = os.path.join(path, 'dqn_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                instance.scaler = pickle.load(f)
    
        return instance
    
    def replay(self, batch_size: int = 32) -> None:
        """经验回放学习"""
        if len(self.memory.buffer) < batch_size:
            return
        
        try:
            # 从缓冲区采样
            samples, indices, weights = self.memory.sample(batch_size)
            logger.debug(f"成功采样{batch_size}个样本")
        
            # 解包经验
            states = np.array([exp[0] for exp in samples])
            actions = np.array([exp[1] for exp in samples])
            rewards = np.array([exp[2] for exp in samples])
            next_states = np.array([exp[3] for exp in samples])
            dones = np.array([exp[4] for exp in samples])
        
            # 计算目标Q值
            targets = self.model.predict(states, verbose=0)
            next_q_values = self.target_model.predict(next_states, verbose=0)
        
            for i in range(batch_size):
                if dones[i]:
                    targets[i][actions[i]] = rewards[i]
                else:
                    targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_q_values[i])
        
            # 更新优先级 (使用TD误差)
            errors = np.abs(targets - self.model.predict(states, verbose=0))
            new_priorities = errors.max(axis=1) + 1e-6  # 添加小常数保证非零
            self.memory.update_priorities(indices, new_priorities)
        
            # 训练模型 (使用重要性采样权重)
            self.model.fit(
                states, 
                targets, 
                sample_weight=weights,
                batch_size=batch_size, 
                verbose=0
            )
        
            # 衰减探索率
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
        except Exception as e:
            logger.error(f"经验回放失败: {str(e)}")
            raise
    
    def generate_signal(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[float]]:
        """生成交易信号"""
        state = self._process_state(df)
        action = self.act(state)
        
        if action == 1:  # buy
            return 'buy', 0.9  # DQN的置信度固定为高
        elif action == 2:  # sell
            return 'sell', 0.9
        return None, None
    
    def _process_state(self, df: pd.DataFrame) -> np.ndarray:
        """
        将市场数据转换为DQN网络输入状态
        参数:
            df: 包含市场数据的DataFrame (至少包含close, volume, 技术指标)
        返回:
            状态向量 (形状: [seq_length, n_features])
        """
        # 1. 选择需要的特征
        features = df[[
            'close', 
            'volume',
            'EMA10', 
            'EMA20',
            'RSI',
            'MACD'
        ]].values
    
        # 2. 标准化处理
        if not hasattr(self, 'scaler'):
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.scaler.fit(features)
    
        # 3. 确保序列长度一致
        seq_length = self.state_shape[0]
        if len(features) > seq_length:
            features = features[-seq_length:]
        elif len(features) < seq_length:
            # 填充序列
            pad_length = seq_length - len(features)
            features = np.pad(features, ((pad_length, 0), (0, 0)), 'edge')
    
        # 4. 标准化并返回
        return self.scaler.transform(features)

    def train(self, env, episodes: int = 1000, batch_size: int = 32, 
             update_target_every: int = 50):
        """
        完整的DQN训练流程
        参数:
            env: 交易环境(需实现reset()和step()方法)
            episodes: 训练轮数
            batch_size: 批大小
            update_target_every: 更新目标网络的频率
        """
        import time
        MAX_STEP_TIME = 60  # 单步最长等待时间(秒)
        train_logger = logging.getLogger('dqn_trainer')

        # 训练统计
        episode_rewards = []
        epsilon_history = []
        
        for episode in range(1, episodes+1):
            logger.info(f"开始第 {episode+1}/{episodes} 回合训练")
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0
        
            while not done:
                steps += 1
                if steps % 100 == 0:  # 每100步记录一次
                    logger.info(f"Episode {episode} - Step {steps} - 当前状态形状: {state.shape}")
                # 1. 选择动作
                action = self.act(state)
            
                # 2. 执行动作
                next_state, reward, done, info = env.step(action)

                # 确保状态形状正确
                if next_state.shape != self.state_shape:
                    logger.error(f"状态形状不匹配! 期望: {self.state_shape}, 实际: {next_state.shape}")
                    raise ValueError("状态形状不匹配")
            
                # 3. 存储经验
                self.remember(state, action, reward, next_state, done)

                # 4. 学习
                self.replay(batch_size)
            
                # 5. 更新状态
                state = next_state
                total_reward += reward
                steps += 1
                if len(self.memory) >= batch_size:
                    self.replay(batch_size)
            
                if steps % update_target_every == 0:
                    self.update_target_network()
        
                # 定期更新目标网络
                if steps % update_target_every == 0:
                    self.update_target_network()

            # 记录统计
            episode_rewards.append(total_reward)
            epsilon_history.append(self.epsilon)

            # 打印进度
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                logger.info(
                    f"Episode {episode}/{episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Epsilon: {self.epsilon:.3f} | "
                    f"Steps: {steps}"
                )

        # 返回训练统计
        return {
            'episode_rewards': episode_rewards,
            'epsilon_history': epsilon_history
        }
