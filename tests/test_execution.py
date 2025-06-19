# tests/test_execution.py

import unittest
from unittest.mock import patch
from core.engine.execution import SmartExecutionEngine

class TestExecutionEngine(unittest.TestCase):
    @patch('core.engine.execution.get_market_depth')
    def test_twap_execution(self, mock_market):
        # 准备测试数据
        mock_market.return_value = {
            'bids': [(1.0800, 1e6), (1.0799, 2e6)],
            'asks': [(1.0801, 1e6), (1.0802, 2e6)]
        }
        
        engine = SmartExecutionEngine({'large_order_threshold': 1e6})
        result = engine.execute_order('EURUSD', 'BUY', 3e6)
        
        # 验证结果
        self.assertTrue(result.success)
        self.assertAlmostEqual(result.execution_price, 1.0801, places=4)
        self.assertLess(result.slippage, 0.0002)
    
    @patch('core.engine.execution.get_market_depth')
    def test_immediate_execution(self, mock_market):
        mock_market.return_value = {
            'bids': [(1.0800, 1e6)],
            'asks': [(1.0801, 1e6)]
        }
        
        engine = SmartExecutionEngine({'large_order_threshold': 1e6})
        result = engine.execute_order('EURUSD', 'BUY', 5e5)
        
        self.assertTrue(result.success)
        self.assertEqual(result.execution_price, 1.0801)
