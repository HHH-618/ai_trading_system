# core/engine/risk.py

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass
from configs.constants import RiskConstants
from configs.hyperparams import RISK_PARAMS
from utils.logger import setup_logger

logger = setup_logger('risk_manager')

@dataclass
class RiskAssessment:
    risk_score: float                   # 综合风险评分(0-1)
    position_size: float                # 建议仓位大小
    allowed_actions: Dict[str, bool]    # 允许的交易动作
    message: str                        # 风险描述

class RiskManager:
    """动态风险管理系统"""
    
    def __init__(self):
        self.portfolio_risk = 0.0
        self.market_risk = 0.0
        self.liquidity_risk = 0.0
        self.position_history = []
        
    def assess_market_risk(self, df: pd.DataFrame) -> RiskAssessment:
        """
        评估市场风险
        :param df: 包含市场数据的DataFrame
        :return: RiskAssessment对象
        """
        if df.empty:
            return RiskAssessment(
                risk_score=1.0,
                position_size=0,
                allowed_actions={'BUY': False, 'SELL': False},
                message="空数据，高风险"
            )
            
        try:
            # 1. 波动率风险
            recent_volatility = df['close'].pct_change().std()
            volatility_risk = min(1.0, recent_volatility / 0.01)  # 假设1%为基准
            
            # 2. 流动性风险
            avg_volume = df['tick_volume'].rolling(20).mean().iloc[-1]
            liquidity_risk = 0 if avg_volume > 1000 else 0.5  # 简化逻辑
            
            # 3. 趋势风险
            adx = df['ADX'].iloc[-1] if 'ADX' in df.columns else 0
            trend_risk = 0.2 if adx > 25 else 0.5
            
            # 4. 相关性风险 (需要多品种数据)
            correlation_risk = 0.3  # 简化
            
            # 综合风险评分
            risk_score = min(1.0, 
                0.4 * volatility_risk + 
                0.3 * trend_risk + 
                0.2 * liquidity_risk + 
                0.1 * correlation_risk
            )
            
            # 动态仓位计算
            position_size = self._calculate_position_size(risk_score)
            
            # 允许的交易动作
            allowed_actions = {
                'BUY': risk_score < 0.7,
                'SELL': risk_score < 0.7
            }
            
            return RiskAssessment(
                risk_score=risk_score,
                position_size=position_size,
                allowed_actions=allowed_actions,
                message=f"风险评分: {risk_score:.2f} (波动率:{volatility_risk:.2f}, 趋势:{trend_risk:.2f})"
            )
            
        except Exception as e:
            logger.error(f"风险评估失败: {str(e)}")
            return RiskAssessment(
                risk_score=1.0,
                position_size=0,
                allowed_actions={'BUY': False, 'SELL': False},
                message=f"风险评估异常: {str(e)}"
            )
            
    def _calculate_position_size(self, risk_score: float) -> float:
        """基于风险评分计算仓位大小"""
        # 风险评分越高，仓位越小
        base_size = RiskConstants.MAX_POSITION_SIZE
        adjusted_size = base_size * (1 - risk_score)
        return max(adjusted_size, 0.01)  # 至少1%
        
    def calculate_value_at_risk(self, returns: pd.Series, alpha: float = 0.95) -> float:
        """计算Value at Risk"""
        if returns.empty:
            return 0.0
        return returns.quantile(1 - alpha)
        
    def check_portfolio_risk(self, positions: Dict[str, float], 
                           correlations: Dict[str, Dict[str, float]]) -> float:
        """
        检查投资组合风险
        :param positions: 当前持仓 {symbol: value}
        :param correlations: 品种相关性矩阵
        :return: 组合风险评分
        """
        if not positions:
            return 0.0
            
        # 简化的组合风险计算
        total_value = sum(positions.values())
        weights = {sym: val/total_value for sym, val in positions.items()}
        
        # 计算组合波动率 (简化版)
        portfolio_vol = 0.0
        for sym1, w1 in weights.items():
            for sym2, w2 in weights.items():
                corr = correlations.get(sym1, {}).get(sym2, 0)
                portfolio_vol += w1 * w2 * corr
                
        return min(1.0, portfolio_vol * 10)  # 缩放至0-1范围
        
    def update_position_history(self, symbol: str, size: float, 
                              entry_price: float, timestamp: pd.Timestamp):
        """更新持仓历史记录"""
        self.position_history.append({
            'symbol': symbol,
            'size': size,
            'entry_price': entry_price,
            'timestamp': timestamp,
            'exit_price': None,
            'exit_time': None
        })

    def calculate_var(self, returns: pd.Series, alpha: float = 0.95, method: str = 'historical') -> float:
        """
        完整的风险价值(VaR)计算
        支持三种方法:
        1. historical - 历史模拟法
        2. parametric - 参数法(正态分布)
        3. monte_carlo - 蒙特卡洛模拟
        """
        if len(returns) < 100:
            logger.warning("数据不足，VaR计算可能不准确")
    
        returns = returns.dropna()
    
        if method == 'historical':
            return self._historical_var(returns, alpha)
        elif method == 'parametric':
            return self._parametric_var(returns, alpha)
        elif method == 'monte_carlo':
            return self._monte_carlo_var(returns, alpha)
        else:
            raise ValueError(f"未知VaR方法: {method}")

    def _historical_var(self, returns: pd.Series, alpha: float) -> float:
        """历史模拟法VaR"""
        return -np.percentile(returns, 100 * (1 - alpha))

    def _parametric_var(self, returns: pd.Series, alpha: float) -> float:
        """参数法VaR(基于正态分布假设)"""
        from scipy.stats import norm
        mu = returns.mean()
        sigma = returns.std()
        return -(mu + sigma * norm.ppf(1 - alpha))

    def _monte_carlo_var(self, returns: pd.Series, alpha: float, n_simulations: int = 10000) -> float:
        """蒙特卡洛模拟法VaR"""
        mu = returns.mean()
        sigma = returns.std()
    
        # 模拟未来收益
        simulated_returns = np.random.normal(mu, sigma, n_simulations)
    
        # 计算VaR
        return -np.percentile(simulated_returns, 100 * (1 - alpha))

    def stress_test(self, positions: Dict[str, float], scenarios: List[Dict[str, float]]) -> Dict:
        """
        执行压力测试
        参数:
            positions: 当前持仓 {symbol: 市值}
            scenarios: 压力场景列表 [{'EURUSD': -0.10, 'XAUUSD': 0.05}, ...]
        返回:
            每个场景下的潜在损失
        """
        results = {}
    
        for scenario in scenarios:
            scenario_loss = 0
            for symbol, pos_value in positions.items():
                if symbol in scenario:
                    # 计算该品种在该场景下的损失
                    shock = scenario[symbol]  # 冲击幅度，如-0.10表示下跌10%
                    scenario_loss += pos_value * shock
        
            results[str(scenario)] = {
                'loss': scenario_loss,
                'loss_pct': scenario_loss / sum(positions.values())
            }
    
        return results

# 单例模式
risk_manager = RiskManager()
