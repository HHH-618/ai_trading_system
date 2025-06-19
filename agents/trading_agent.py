# agents/trading_agent.py

from typing import Dict, Any
from core.engine.execution import ExecutionEngine
from core.engine.risk import RiskManager
from core.models.meta_learner import MetaLearner
from core.strategies.rl.dqn import DQNStrategy

class TradingAgent:
    """智能交易代理，协调策略执行"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.execution_engine = ExecutionEngine(config)
        self.risk_manager = RiskManager(config)
        self.strategies = self._init_strategies()
        self.meta_learner = MetaLearner(input_shape=(60, 20))  # 示例形状
        
    def _init_strategies(self) -> Dict[str, Any]:
        """初始化策略集合"""
        return {
            'dqn': DQNStrategy(state_shape=(60, 20)),
            # 可以添加更多策略
        }
    
    def run_cycle(self, market_data: Dict[str, Any]) -> None:
        """运行交易周期"""
        # 1. 风险评估
        risk_assessment = self.risk_manager.assess(market_data)
        
        if not risk_assessment['trade_allowed']:
            return
            
        # 2. 元学习适应
        self.meta_learner.adapt(market_data['features'], market_data['target'])
        
        # 3. 策略决策
        for name, strategy in self.strategies.items():
            signal, confidence = strategy.generate_signal(market_data)
            
            if signal and confidence > 0.7:  # 置信度阈值
                # 4. 风险管理
                position_size = self.risk_manager.calculate_position_size(
                    market_data['price'], 
                    risk_assessment['risk_score']
                )
                
                # 5. 执行交易
                self.execution_engine.execute(
                    symbol=market_data['symbol'],
                    signal=signal,
                    size=position_size,
                    strategy=name
                )
                
    def learn_from_experience(self, experiences: Dict[str, Any]) -> None:
        """从交易经验中学习"""
        # 更新强化学习策略
        self.strategies['dqn'].remember(
            experiences['state'],
            experiences['action'],
            experiences['reward'],
            experiences['next_state'],
            experiences['done']
        )
        self.strategies['dqn'].replay()
        
        # 更新元学习器
        self.meta_learner.adapt(
            np.array([experiences['state']]),
            np.array([experiences['meta_target']])
        )
