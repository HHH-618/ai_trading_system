# AI Trading System

![System Architecture](docs/architecture.png)

专业级外汇黄金量化交易系统，整合机器学习与算法交易技术。

## 主要特性

- **多策略融合**: 结合监督学习、强化学习和元学习
- **智能执行**: VWAP/TWAP算法、冰山订单支持
- **动态风控**: 实时风险价值(VaR)计算、压力测试
- **自适应学习**: 在线学习、模型自动调优
- **多时间框架**: 支持M1到D1级别的分析

## 系统架构
ai_trading_system/
│── configs/
│   ├── __init__.py
│   ├── settings.py           # 动态配置管理
│   ├── constants.py          # 常量定义
│   └── hyperparams.py        # 超参数配置
│── core/
│   ├── data/                 # 数据模块
│   │   ├── fetcher.py        # 多源数据获取
│   │   ├── processor.py      # 高级数据处理
│   │   ├── storage.py        # 数据存储管理
│   │   └── streaming.py      # 实时数据流处理
│   ├── engine/               # 交易引擎
│   │   ├── execution.py      # 智能订单执行
│   │   ├── risk.py           # 动态风险管理
│   │   └── portfolio.py      # 组合管理
│   ├── models/               # AI模型
│   │   ├── meta_learner.py   # 元学习框架
│   │   ├── ensemble.py       # 模型集成
│   │   ├── trainer.py        # 自适应训练
│   │   └── optimizer.py      # 超参数优化
│   ├── strategies/           # 策略模块
│   │   ├── base.py           # 策略基类
│   │   ├── rl/               # 强化学习策略
│   │   │   ├── environment.py        # 交易环境类
│   │   │   ├── dqn.py        # DQN策略
│   │   │   └── ppo.py        # PPO策略
│   │   └── supervised/       # 监督学习策略
│   │       ├── lstm.py       # LSTM策略
│   │       └── transformer.py # Transformer策略
│   └── utils/                # 核心工具
│       ├── memory.py         # 经验回放
│       └── scaler.py         # 自适应标准化
│── research/                 # 研究模块
│   ├── notebooks/            # Jupyter notebooks
│   ├── experiments/          # 实验脚本
│   └── analysis/             # 结果分析
│── utils/
│   ├── logger.py             # 增强日志
│   ├── visualizer.py         # 可视化工具
│   ├── scheduler.py          # 任务调度
│   └── helpers.py            # 实用函数
│── agents/                   # 智能体
│   ├── trading_agent.py      # 交易智能体
│   └── learning_agent.py     # 学习智能体
│── scripts/                  # 实用脚本
│   ├── deploy.py             # 部署脚本
│   └── monitor.py            # 监控脚本
│── backtest/                 # 回测模块
│   ├── walkforward.py        # 步进式回测
│   └── optimizer.py          # 参数优化
│── tests/
│   ├── test_risk.py          #风险管理测试
│   └── test_execution.py          # 单元测试
│── app.py                    # 主应用入口
│── train.py                  # 训练入口
│── trade.py                  # 交易入口
│── requirements.txt          # 依赖文件
└── README.md

## 快速开始
1. 创建虚拟环境
conda create --name AI python=3.9
2. 进入虚拟环境
conda activate AI
3. 安装依赖文件
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
pip install numpy pandas scikit-learn tensorflow keras statsmodels MetaTrader5 ccxt gym stable-baselines3 python-dotenv schedule tqdm prometheus-client docker fabric matplotlib seaborn plotly -i https://mirrors.aliyun.com/pypi/simple/
conda install -c conda-forge ta-lib
4. 进入项目文件
cd ai_trading_system
5. 开始训练模型
python train.py --symbol XAUUSD.r --timeframe H1
6. 启动实盘交易
python trade.py --mode live
7. 查看监控系统  
python scripts/monitor.py
