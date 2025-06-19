# core/engine/execution.py

import time
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from core.utils.logger import setup_logger
from configs.settings import Config

logger = setup_logger('execution')

@dataclass
class ExecutionResult:
    success: bool
    order_id: Optional[str]
    execution_price: Optional[float]
    slippage: Optional[float]
    message: str

class SmartExecutionEngine:
    """智能订单执行引擎，包含算法交易逻辑"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.pending_orders = []
        self.vwap_cache = {}
        
    def execute_order(self, symbol: str, side: str, quantity: float, 
                     strategy: str = 'default') -> ExecutionResult:
        """智能订单执行主逻辑"""
        
        # 1. 获取市场深度
        market_depth = self._get_market_depth(symbol)
        if not market_depth:
            return ExecutionResult(False, None, None, None, "无法获取市场深度")
        
        # 2. 计算最优执行参数
        exec_params = self._calculate_execution_params(
            symbol, side, quantity, market_depth)
        
        # 3. 执行算法
        if quantity >= self.config['large_order_threshold']:
            # 大单拆分执行
            return self._execute_twap(symbol, side, quantity, exec_params)
        else:
            # 普通订单
            return self._execute_immediate(symbol, side, quantity, exec_params)
    
    def _calculate_execution_params(self, symbol: str, side: str, 
                                 quantity: float, market_depth: Dict) -> Dict:
        """
        计算最优执行参数
        包括:
        - 执行算法选择
        - 执行时间段
        - 子订单大小
        - 允许的最大滑点
        """
        params = {}
    
        # 1. 根据订单大小选择算法
        large_order_threshold = self.config.get('large_order_threshold', 1000000)  # 默认100万单位
        if quantity >= large_order_threshold:
            params['algorithm'] = 'TWAP'
        else:
            params['algorithm'] = 'IMMEDIATE'
    
        # 2. 计算市场流动性
        liquidity_score = self._calculate_liquidity_score(market_depth)
    
        # 3. 动态调整执行参数
        if params['algorithm'] == 'TWAP':
            # TWAP参数
            params['duration'] = max(5, min(60, quantity / (liquidity_score * 10000)))  # 5-60分钟
            params['chunks'] = max(3, int(params['duration'] / 5))  # 每5分钟一个子订单
            params['max_slippage'] = 0.002  # 0.2%
        else:
            # 即时执行参数
            params['max_slippage'] = 0.001  # 0.1%
    
        # 4. 考虑市场波动性
        if self._is_high_volatility(symbol):
            params['max_slippage'] *= 1.5  # 高波动市场允许更大滑点
    
        return params

    def _calculate_liquidity_score(self, market_depth: Dict) -> float:
        """计算市场流动性得分"""
        # 计算前3档的深度
        bid_liquidity = sum(vol for price, vol in market_depth['bids'][:3])
        ask_liquidity = sum(vol for price, vol in market_depth['asks'][:3])
        return (bid_liquidity + ask_liquidity) / 2

    def _is_high_volatility(self, symbol: str) -> bool:
        """检查当前市场是否高波动"""
        # 这里可以接入实时波动率数据
        # 简化实现：检查最近5分钟的波动率
        return False  # 实际实现需要接入真实数据
    
    def _execute_twap(self, symbol: str, side: str, quantity: float, params: Dict) -> ExecutionResult:
        """实现完整TWAP算法"""
        duration_minutes = params.get('duration', 30)  # 默认30分钟执行期
        intervals = max(1, duration_minutes // 5)  # 每5分钟一个子订单
    
        filled = 0
        avg_price = 0
        start_time = time.time()
    
        for i in range(intervals):
            # 等待到下一个执行点
            sleep_time = start_time + (i+1)*300 - time.time()  # 300秒=5分钟
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # 获取当前市场状态
            market_data = self._get_market_depth(symbol)
            if not market_data:
                return ExecutionResult(False, None, None, f"无法获取市场数据-区间{i+1}")
            
            # 计算本次执行量 (按时间等分)
            chunk = min(quantity - filled, quantity / intervals)
        
            # 执行子订单
            result = self._execute_chunk(symbol, side, chunk, market_data)
            if not result.success:
                logger.warning(f"TWAP部分执行失败: {result.message}")
                continue
            
            # 更新统计
            filled += result.executed_quantity
            avg_price = (avg_price * (filled - result.executed_quantity) + 
                    result.execution_price * result.executed_quantity) / filled
    
        return ExecutionResult(
            success=filled > 0,
            order_id=f"TWAP-{int(start_time)}",
            execution_price=avg_price,
            slippage=(avg_price - self._get_benchmark_price(side, start_time)) / avg_price,
            message=f"TWAP完成 {filled}/{quantity}"
        )

    def _execute_chunk(self, symbol: str, side: str, quantity: float, market_data: Dict):
        """执行单个子订单"""
        # 根据市场深度计算最优价格
        price_levels = market_data['asks'] if side == 'BUY' else market_data['bids']
        remaining = quantity
        executed = 0
        avg_price = 0
    
        for price, volume in price_levels:
            take = min(remaining, volume)
            if take <= 0:
                break
            
            executed += take
            avg_price = (avg_price * (executed - take) + price * take) / executed
            remaining -= take
    
        slippage = self._calculate_slippage(side, price_levels[0][0], avg_price)
    
        return ExecutionResult(
            success=executed > 0,
            order_id=f"CHUNK-{time.time()}",
            execution_price=avg_price,
            slippage=slippage,
            executed_quantity=executed,
            message=f"执行{executed}@{avg_price:.5f}"
        )
    
    def _execute_immediate(self, symbol: str, side: str, quantity: float,
                     params: Dict) -> ExecutionResult:
        """
        即时执行订单的完整实现
        参数:
            symbol: 交易品种 (如'EURUSD')
            side: 买卖方向 ('BUY'/'SELL')
            quantity: 交易量
            params: 执行参数 (如允许的最大滑点)
        返回:
            ExecutionResult 对象
        """
        try:
            # 1. 获取当前市场深度
            market_depth = self._get_market_depth(symbol)
            if not market_depth:
                return ExecutionResult(False, None, None, None, "无法获取市场深度")
        
            # 2. 根据订单方向选择市场深度
            price_levels = market_depth['asks'] if side == 'BUY' else market_depth['bids']
        
            # 3. 计算可执行量
            remaining = quantity
            executed = 0
            avg_price = 0
            slippage = 0
        
            # 4. 遍历市场深度执行订单
            for price, volume in price_levels:
                if remaining <= 0:
                    break
                
                # 计算当前价位可执行量
                take = min(remaining, volume)
                executed += take
                avg_price = (avg_price * (executed - take) + price * take) / executed
                remaining -= take
        
            # 5. 计算实际滑点
            first_price = price_levels[0][0]
            slippage = abs(avg_price - first_price) / first_price
        
            # 6. 检查是否超过允许的最大滑点
            max_slippage = params.get('max_slippage', 0.001)  # 默认0.1%
            if slippage > max_slippage:
                return ExecutionResult(
                    False, None, None, None, 
                    f"滑点{slippage:.2%}超过最大允许值{max_slippage:.2%}"
                )
        
            # 7. 生成订单ID并返回结果
            order_id = f"IMM-{int(time.time())}"
            return ExecutionResult(
                success=True,
                order_id=order_id,
                execution_price=avg_price,
                slippage=slippage,
                message=f"即时订单执行成功 {executed}/{quantity} @ {avg_price:.5f}"
            )
        
        except Exception as e:
            logger.error(f"即时订单执行失败: {str(e)}")
            return ExecutionResult(False, None, None, None, f"执行异常: {str(e)}")
