# core/strategies/rl/ppo.py

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from core.utils.memory import PPOMemory

class PPONetwork(nn.Module):
    """PPO策略网络"""
    
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super().__init__()
        self.flatten = nn.Flatten()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, output_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.flatten(1)
        shared_out = self.shared(x)
        return F.softmax(self.actor(shared_out), self.critic(shared_out))

class PPOStrategy:
    """近端策略优化(PPO)策略"""
    
    def __init__(self, input_dim, action_dim, lr=3e-4, gamma=0.99, 
                clip_param=0.2, update_epochs=3, batch_size=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PPONetwork(input_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.memory = PPOMemory()
        self.gamma = gamma
        self.clip_param = clip_param
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.action_dim = action_dim
        
    def act(self, state):
        """选择动作"""
        state = torch.FloatTensor(state).to(self.device)
        state = state.flatten() 
        with torch.no_grad():
            probs, value = self.network(state)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item(), value.item()
        
    def update(self):
        """PPO更新步骤"""
        states, actions, old_log_probs, rewards, dones, values = self.memory.sample()
        
        # 计算折扣回报和优势函数
        returns = self._compute_returns(rewards, dones, values)
        advantages = returns - values
        
        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)
        returns = torch.FloatTensor(np.array(returns)).to(self.device)
        advantages = torch.FloatTensor(np.array(advantages)).to(self.device)
        
        # 多轮更新
        for _ in range(self.update_epochs):
            # 获取新概率和状态值
            new_probs, new_values = self.network(states)
            dist = Categorical(new_probs)
            new_log_probs = dist.log_prob(actions)
            
            # 计算比率
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # 计算损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip_param, 1+self.clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = F.mse_loss(new_values.squeeze(), returns)
            
            entropy_loss = dist.entropy().mean()
            
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()

    def _compute_returns(self, rewards, dones, values):
        """计算折扣回报"""
        returns = np.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns[t] = running_return
            
        return returns
    
    def train(self, env, episodes=1000):
        """完整训练流程"""
        logger = logging.getLogger('ppo_trainer')
        episode_rewards = []
        
        for episode in range(1, episodes+1):
            state = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # 选择动作
                action, log_prob, value = self.act(state)
                
                # 执行动作
                next_state, reward, done, info = env.step(action)
                
                # 存储经验
                self.memory.add(state, action, log_prob, reward, done, value)
                
                # 更新状态
                state = next_state
                total_reward += reward
                
                # 定期更新策略
                if len(self.memory) >= self.batch_size:
                    self.update()
                    self.memory.clear()
            
            # 记录统计
            episode_rewards.append(total_reward)
            
            # 打印进度
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                logger.info(
                    f"Episode {episode}/{episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Total Reward: {total_reward:.2f}"
                )
        
        return episode_rewards
