# tests/test_risk.py

import unittest
import pandas as pd
import numpy as np
from core.engine.risk import RiskManager

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.risk_mgr = RiskManager()
        
    def test_var_calculation(self):
        # 生成测试数据
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0001, 0.01, 1000))
        
        # 测试历史法VaR
        historical_var = self.risk_mgr.calculate_var(returns, method='historical')
        self.assertAlmostEqual(historical_var, 0.016, places=3)
        
        # 测试参数法VaR
        parametric_var = self.risk_mgr.calculate_var(returns, method='parametric')
        self.assertAlmostEqual(parametric_var, 0.016, places=3)
    
    def test_stress_test(self):
        positions = {'EURUSD': 1e6, 'XAUUSD': 5e5}
        scenarios = [
            {'EURUSD': -0.05, 'XAUUSD': 0.03},  # 欧元下跌5%，黄金上涨3%
            {'EURUSD': -0.10, 'XAUUSD': 0.10}   # 欧元下跌10%，黄金上涨10%
        ]
        
        results = self.risk_mgr.stress_test(positions, scenarios)
        
        self.assertAlmostEqual(results[str(scenarios[0])]['loss'], -1e6*0.05 + 5e5*0.03)
        self.assertAlmostEqual(results[str(scenarios[1])]['loss_pct'], (-1e6*0.10 + 5e5*0.10)/1.5e6)
