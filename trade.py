# trade.py

import logging
import time
from typing import Dict, Any
from core.data.streaming import DataStreamer
from agents.trading_agent import TradingAgent
from utils.helpers import calculate_sharpe_ratio
from configs.settings import Config

class LiveTrader:
    """实时交易执行器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.symbol = config['trading']['symbols'][0]
        self.timeframe = config['trading']['timeframes'][0]
        self.agent = TradingAgent(config)
        self.streamer = DataStreamer(self.symbol, [self.timeframe])
        self.performance_metrics = {
            'win_rate': 0,
            'sharpe': 0,
            'drawdown': 0
        }
        
    def start(self):
        """启动交易循环"""
        logger = logging.getLogger('live_trader')
        logger.info(f"开始{symbol}的实时交易")
        
        # 启动数据流
        self.streamer.start()
        
        try:
            while True:
                # 获取最新市场数据
                market_data = self._collect_market_data()
                
                # 运行交易周期
                self.agent.run_cycle(market_data)
                
                # 更新性能指标
                self._update_performance()
                
                # 控制循环频率
                time.sleep(60)  # 每分钟运行一次
                
        except KeyboardInterrupt:
            logger.info("交易安全停止")
        except Exception as e:
            logger.error(f"交易异常: {e}", exc_info=True)
            raise
            
    def _collect_market_data(self) -> Dict[str, Any]:
        """收集并处理市场数据"""
        latest_candle = self.streamer.get_latest(self.timeframe, n=1)[0]
        
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'price': (latest_candle['bid'] + latest_candle['ask']) / 2,
            'features': latest_candle,
            'timestamp': datetime.now()
        }
        
    def _update_performance(self):
        """更新交易绩效指标"""
        # 这里需要连接实际交易记录进行计算
        pass

if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger('live_trader')
    
    trader = LiveTrader(Config.load_config())
    trader.start()
