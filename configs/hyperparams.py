# configs/hyperparams.py

"""
模型超参数配置
"""

# 元学习器超参数
META_LEARNER_PARAMS = {
    'input_dim': 64,
    'hidden_dim': 128,
    'output_dim': 3,
    'learning_rate': 0.001,
    'adaptation_steps': 3,
    'meta_batch_size': 32,
    'clip_param': 0.2
}

# DQN超参数
DQN_PARAMS = {
    'gamma': 0.99,                # 折扣因子
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'learning_rate': 0.0005,
    'target_update': 100,         # 目标网络更新频率
    'memory_capacity': 10000,
    'batch_size': 64
}

# PPO超参数
PPO_PARAMS = {
    'gamma': 0.99,
    'lam': 0.95,                  # GAE参数
    'clip_param': 0.2,
    'entropy_coef': 0.01,
    'learning_rate': 0.0003,
    'update_epochs': 4,
    'mini_batch_size': 64
}

# 数据预处理参数
DATA_PARAMS = {
    'lookback_window': 60,
    'train_test_split': 0.8,
    'feature_groups': {
        'price': ['open', 'high', 'low', 'close'],
        'volume': ['tick_volume', 'real_volume'],
        'indicators': ['EMA10', 'EMA20', 'RSI', 'MACD']
    },
    'normalization': 'robust'     # robust/minmax/zscore
}

# 风险模型参数
RISK_PARAMS = {
    'volatility_lookback': 20,     # 波动率计算窗口
    'correlation_window': 60,      # 相关性计算窗口
    'value_at_risk_alpha': 0.95,   # VaR置信度
    'max_drawdown_threshold': 0.2  # 最大回撤阈值
}
