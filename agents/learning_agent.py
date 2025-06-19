# agents/learning_agent.py

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from core.models.meta_learner import MetaLearner
from core.models.ensemble import ModelEnsemble
from core.utils.memory import PrioritizedReplayBuffer
from utils.logger import setup_logger

logger = setup_logger('learning_agent')

class LearningAgent:
    """
    学习智能体 - 负责持续学习和模型优化
    功能:
    1. 在线学习
    2. 经验回放
    3. 模型再训练
    4. 性能监控
    """
    
    def __init__(self, 
                 initial_models: List,
                 state_dim: int,
                 action_dim: int = 3):  # buy, sell, hold
        self.meta_learner = MetaLearner(input_shape=(state_dim,))
        self.model_ensemble = ModelEnsemble(initial_models)
        self.memory = PrioritizedReplayBuffer(capacity=10000)
        self.learning_iterations = 0
        self.performance_history = []
        
    def learn_from_experience(self, 
                            state: np.ndarray,
                            action: int,
                            reward: float,
                            next_state: np.ndarray,
                            done: bool,
                            importance: float = 1.0):
        """
        从单次经验中学习
        :param importance: 经验重要性权重
        """
        self.memory.add(state, action, reward, next_state, done, importance)
        
        # 每隔一定步数进行批量学习
        if len(self.memory) >= 64 and self.learning_iterations % 10 == 0:
            self.replay_experience(batch_size=64)
            
        self.learning_iterations += 1
        
    def replay_experience(self, batch_size: int = 32):
        """从记忆中回放学习"""
        batch, indices, weights = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = batch
        
        # 1. 更新元学习器
        self.meta_learner.adapt(states, rewards)
        
        # 2. 更新集成模型
        for i, model in enumerate(self.model_ensemble.models):
            if hasattr(model, 'train_on_batch'):  # 如果是Keras模型
                loss = model.train_on_batch(states, actions, sample_weight=weights)
                logger.debug(f"模型{i} 更新 - 损失: {loss:.4f}")
        
        # 3. 更新优先级
        new_priorities = self._calculate_priorities(states, actions)
        self.memory.update_priorities(indices, new_priorities)
        
    def _calculate_priorities(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """基于TD误差计算新的优先级"""
        # 这里使用简单的预测误差作为优先级
        predictions = self.model_ensemble.weighted_predict(states)
        errors = np.abs(predictions - actions.reshape(-1, 1))
        return np.clip(errors.flatten(), 1e-6, 10)  # 防止优先级为0
        
    def evaluate_performance(self, 
                           test_data: pd.DataFrame,
                           metrics: List[str] = ['accuracy', 'sharpe']) -> Dict[str, float]:
        """评估当前模型性能"""
        results = {}
        
        if 'accuracy' in metrics:
            # 计算预测准确率
            pass
            
        if 'sharpe' in metrics:
            # 计算模拟交易的夏普比率
            pass
            
        self.performance_history.append(results)
        return results
        
    def adaptive_model_update(self, new_performance: Dict[str, float]):
        """
        根据最新性能调整模型权重
        :param new_performance: 最新性能指标
        """
        if not self.performance_history:
            return
            
        # 计算性能变化趋势
        last_perf = self.performance_history[-1]
        improvements = {
            k: new_performance[k] - last_perf[k] 
            for k in new_performance if k in last_perf
        }
        
        # 调整模型权重
        self.model_ensemble.update_weights(list(improvements.values()))
        
    def save_models(self, dir_path: str):
        """保存所有模型和记忆"""
        import os
        os.makedirs(dir_path, exist_ok=True)
        
        self.meta_learner.save(f"{dir_path}/meta_learner.h5")
        self.model_ensemble.save_ensemble(f"{dir_path}/ensemble")
        self.memory.save(f"{dir_path}/memory.pkl")
        
    def load_models(self, dir_path: str):
        """加载模型和记忆"""
        self.meta_learner.load(f"{dir_path}/meta_learner.h5")
        self.model_ensemble = ModelEnsemble.load_ensemble(f"{dir_path}/ensemble")
        self.memory.load(f"{dir_path}/memory.pkl")
        
    def periodic_retrain(self, new_data: pd.DataFrame):
        """
        周期性再训练模型
        :param new_data: 新收集的数据
        """
        # 1. 数据预处理
        # ...
        
        # 2. 微调模型
        for model in self.model_ensemble.models:
            if hasattr(model, 'fit'):
                model.fit(new_data, epochs=5, verbose=0)
                
        logger.info("模型周期性再训练完成")
