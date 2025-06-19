# core/utils/memory.py

import random
import numpy as np
from collections import deque
from typing import NamedTuple

class Experience(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float = None
    value: float = None

class ReplayBuffer:
    """标准经验回放缓冲区"""
    
    def __init__(self, capacity=10000, alpha=0.6):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        # 初始优先级设为1.0 + 小常数
        self.initial_priority = 1.0 + 1e-6
        
    def add(self, experience: Experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> list:
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    
    def __len__(self):
        return len(self.buffer)

class PrioritizedReplayBuffer(ReplayBuffer):
    """带优先级的经验回放"""
    
    def __init__(self, capacity=10000, alpha=0.6):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        
    def add(self, experience, priority=1.0):
        """添加经验到缓冲区"""
        self.buffer.append(experience)
        self.priorities.append(priority ** self.alpha)
        
    def sample(self, batch_size: int, beta: float = 0.4) -> tuple:
        # 计算采样概率
        priorities = np.array(self.priorities)
        probs = priorities / (priorities.sum() + 1e-8)  # 添加小常数防止除零
    
        # 检查概率总和是否接近1
        if not np.isclose(probs.sum(), 1.0, atol=1e-3):
            probs = np.ones_like(probs) / len(probs)  # 如果概率不正常，使用均匀分布
    
        # 采样
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # 归一化重要性采样权重
    
        return samples, indices, weights
    
    def update_priorities(self, indices: list, priorities: list):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):  # 确保索引有效
                self.priorities[idx] = (abs(priority) + 1e-6) ** self.alpha  # 保证非负
            
class PPOMemory:
    """PPO专用记忆缓冲区"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
    def add(self, state, action, log_prob, reward, done, value):
        """添加经验"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        
    def sample(self):
        """获取所有经验"""
        return (
            self.states,
            self.actions,
            self.log_probs,
            self.rewards,
            self.dones,
            self.values
        )
        
    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
    def __len__(self):
        return len(self.states)
