# train.py

import os
import sys
import logging
import torch
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from core.data.fetcher import data_fetcher
from core.data.processor import data_processor
from core.models.trainer import AdaptiveTrainer
from core.strategies.supervised.lstm import LSTMStrategy
from core.strategies.supervised.transformer import TransformerStrategy
from core.strategies.rl.dqn import DQNStrategy
from core.strategies.rl.ppo import PPOStrategy
from utils.visualizer import TradingVisualizer
from configs.settings import Config
from configs.constants import ModelConstants
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data(symbol: str, timeframe: str, days: int = 365) -> tuple:
    """准备训练数据"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # 获取原始数据
    raw_data = data_fetcher.get_historical_data(
        symbol=symbol,
        timeframe=timeframe,
        from_date=start_date
    )
    
    # 计算技术指标
    processed_data = data_processor.calculate_technical_indicators(raw_data)
    
    # 标准化处理
    normalized_data = data_processor.normalize_data(processed_data)

    # 创建目标变量 (预测下一期的收盘价)
    y = normalized_data['close'].shift(-1).dropna().values
    # 创建两个目标变量
    close_prices = normalized_data['close'].values
    future_prices = normalized_data['close'].shift(-1).dropna().values
    
    # 方向目标 (1=上涨, 0=下跌)
    direction = (future_prices > close_prices[:len(future_prices)]).astype(int)
    
    # 幅度目标 (价格变化百分比)
    magnitude = (future_prices - close_prices[:len(future_prices)]) / close_prices[:len(future_prices)]
    
    # 确保X和y长度一致
    normalized_data = normalized_data.iloc[:len(direction)]
    
    # 转换为序列数据
    X, y_direction, y_magnitude = create_sequences(
        data=normalized_data.values,
        direction_targets=direction,
        magnitude_targets=magnitude,
        seq_length=ModelConstants.SEQ_LENGTH
    )

    y_direction = y_direction.reshape(-1, 1)  # 从(2829,)变为(2829,1)
    y_magnitude = y_magnitude.reshape(-1, 1)  # 从(2829,)变为(2829,1)
    
    return X, {
        'direction': y_direction.astype(np.float32),
        'magnitude': y_magnitude.astype(np.float32)
    }

def create_sequences(data: np.ndarray, 
                   direction_targets: np.ndarray, 
                   magnitude_targets: np.ndarray, 
                   seq_length: int) -> tuple:
    """将数据组织成序列形式"""
    X, y_dir, y_mag = [], [], []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y_dir.append(direction_targets[i+seq_length-1])
        y_mag.append(magnitude_targets[i+seq_length-1])
        
    return np.array(X), np.array(y_dir), np.array(y_mag)

def prepare_dqn_data(symbol, timeframe, days=180):
    """准备DQN训练数据"""
    # 获取历史数据
    logger.info("准备DQN训练数据")
    raw_data = data_fetcher.get_historical_data(
        symbol=symbol,
        timeframe=timeframe,
        from_date=datetime.now() - timedelta(days=days)
    )
    logger.info(f"获取到原始数据，长度: {len(raw_data)}")
    
    # 处理数据
    processed_data = data_processor.calculate_technical_indicators(raw_data)
    logger.info("技术指标计算完成")
    normalized_data = data_processor.normalize_data(processed_data)
    logger.info(f"数据标准化完成，最终形状: {normalized_data.shape}")
    
    return normalized_data

def run_dqn_training(config):
    """运行DQN训练"""
    logger.info("开始DQN训练流程")
    
    # 1. 准备数据
    logger.info("准备数据")
    data = prepare_dqn_data(
        symbol=config['trading']['symbols'][0],
        timeframe='H1'
    )
    logger.info(f"数据准备完成，形状: {data.shape}")
    
    # 2. 创建交易环境
    logger.info("创建交易环境...")
    from core.strategies.rl.environment import TradingEnvironment
    env = TradingEnvironment(data)
    logger.info(f"环境初始化完成，最大步数: {env.max_steps}")
    
    # 3. 初始化DQN策略
    logger.info("初始化DQN模型...")
    state_shape = (ModelConstants.SEQ_LENGTH, data.shape[1])
    dqn_model = DQNStrategy(state_shape=state_shape)
    logger.info(f"模型初始化完成，状态形状: {state_shape}")
    
    # 4. 训练模型
    logger.info("开始训练...")
    start_time = time.time()
    episodes = 1
    with tqdm(total=episodes, desc="DQN训练进度") as pbar:
        training_stats = dqn_model.train(
            env=env,
            episodes=episodes,
            batch_size=128,#64
            update_target_every=100
        )
        pbar.update(episodes)

    training_time = time.time() - start_time
    logger.info(f"DQN训练完成，耗时: {training_time/60:.2f}分钟")
    
    # 5. 保存模型
    logger.info("保存模型...")
    dqn_model.save(str(Config.MODEL_DIR / 'dqn_model'))
    logger.info("DQN模型训练完成并保存")
    
    return training_stats

def run_ppo_training(config):
    """运行PPO训练"""
    logger.info("开始PPO训练流程")
    
    # 1. 准备数据
    data = prepare_dqn_data(
        symbol=config['trading']['symbols'][0],
        timeframe='H1'
    )
    
    # 2. 创建交易环境
    from core.strategies.rl.environment import TradingEnvironment
    env = TradingEnvironment(data)
    
    # 3. 初始化PPO策略
    state_dim = data.shape[1] * ModelConstants.SEQ_LENGTH
    ppo_model = PPOStrategy(
        input_dim=state_dim,
        action_dim=3,  # 买、卖、持有
        lr=3e-4,
        batch_size=128,
        update_epochs=3
    )

    # 4. 训练模型 - 添加进度条
    start_time = time.time()
    episodes = 10
    with tqdm(total=episodes, desc="PPO训练进度") as pbar:
        training_stats = ppo_model.train(
            env=env,
            episodes=episodes
        )

    training_time = time.time() - start_time
    logger.info(f"PPO训练完成，耗时: {training_time/60:.2f}分钟")
    
    # 5. 保存模型
    torch.save(ppo_model.network.state_dict(), Config.MODEL_DIR / 'ppo_model.pth')
    logger.info("PPO模型训练完成并保存")
    
    return training_stats

def plot_training_results(dqn_stats, ppo_stats):
    """可视化训练结果"""
    plt.figure(figsize=(12, 6))
    
    # DQN训练奖励
    plt.subplot(1, 2, 1)
    plt.plot(dqn_stats['episode_rewards'])
    plt.title("DQN训练奖励")
    plt.xlabel("Episode")
    plt.ylabel("奖励")
    
    # PPO训练奖励
    plt.subplot(1, 2, 2)
    plt.plot(ppo_stats)
    plt.title("PPO训练奖励")
    plt.xlabel("Episode")
    plt.ylabel("奖励")
    
    plt.tight_layout()
    plt.savefig(Config.MODEL_DIR / 'training_results.png')
    plt.close()

def run_training(config: dict):
    """执行完整训练流程"""
    logger = logging.getLogger('train')
    logger.info("开始模型训练流程")
    
    # 1. 数据准备
    logger.info("准备训练数据...")
    X, y_dict = prepare_data(
        symbol=config['trading']['symbols'][0],
        timeframe='H1',
        days=180
    )
    logger.info(f"序列数据形状 - X: {X.shape}")
    logger.info(f"方向目标形状: {y_dict['direction'].shape}")
    logger.info(f"幅度目标形状: {y_dict['magnitude'].shape}")

    # 手动分割训练集和验证集
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train = {'direction': y_dict['direction'][:split_idx], 'magnitude': y_dict['magnitude'][:split_idx]}
    y_val = {'direction': y_dict['direction'][split_idx:], 'magnitude': y_dict['magnitude'][split_idx:]}
    
    logger.info(f"训练数据形状 - X: {X_train.shape}, direction: {len(y_train['direction'])}, magnitude: {len(y_train['magnitude'])}")
    logger.info(f"验证数据形状 - X: {X_val.shape}, direction: {len(y_val['direction'])}, magnitude: {len(y_val['magnitude'])}")
    
    # 2. 训练LSTM模型
    lstm_model = LSTMStrategy()
    lstm_model.model = lstm_model.build_model(input_shape=X_train.shape[1:])

    logger.info("训练LSTM模型...")
    lstm_trainer = AdaptiveTrainer(lstm_model.model)
    lstm_history = lstm_trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,  # 显式传入验证数据
        epochs=100
    )

    # 保存LSTM模型和标准化器
    logger.info("保存训练好的LSTM模型...")
    lstm_model.scaler = data_processor.scalers['price']
    lstm_model.model.save(Config.MODEL_DIR / 'lstm_model.keras')

    # 3. 训练Transformer模型
    transformer_model = TransformerStrategy()
    transformer_model.model = transformer_model.build_model(input_shape=X_train.shape[1:])

    logger.info("训练Transformer模型...")
    transformer_trainer = AdaptiveTrainer(transformer_model.model)
    transformer_history = transformer_trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=50
    )

    # 保存Transformer模型
    logger.info("保存训练好的Transformer模型...")
    transformer_model.model.save(Config.MODEL_DIR / 'transformer_model.keras')

    # 4. 训练强化学习模型
    dqn_stats = run_dqn_training(config)
    ppo_stats = run_ppo_training(config)
    
    # 可视化训练结果
    plot_training_results(dqn_stats, ppo_stats)
        
    logger.info("所有模型训练完成")

if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger('train')
    run_training(Config.load_config())
