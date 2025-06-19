# app.py

import argparse
import logging
from utils.logger import setup_logger
from utils.scheduler import TradingScheduler
from scripts.monitor import TradingSystemMonitor
from agents.trading_agent import TradingAgent
from configs.settings import Config

def main():
    # 配置日志
    setup_logger('main', level=logging.INFO)
    logger = logging.getLogger('main')
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='AI交易系统主程序')
    parser.add_argument('--mode', choices=['train', 'trade', 'monitor', 'all'], 
                       default='all', help='运行模式')
    parser.add_argument('--config', default='config/main_config.yaml', 
                       help='配置文件路径')
    args = parser.parse_args()

    # 加载配置
    config = Config.load_config()
    logger.info("系统配置加载完成")

    try:
        # 模式分发
        if args.mode in ['train', 'all']:
            from train import run_training
            run_training(config)
            
        if args.mode in ['trade', 'all']:
            trading_agent = TradingAgent(config)
            scheduler = TradingScheduler()
            
            # 添加定时任务
            scheduler.add_intraday_job(
                trading_agent.run_cycle,
                interval_minutes=5,
                kwargs={'market_data': 'auto'}
            )
            
            # 启动调度器
            scheduler.start()
            logger.info("交易调度器已启动")
            
        if args.mode in ['monitor', 'all']:
            monitor = TradingSystemMonitor()
            monitor.run()
            
    except KeyboardInterrupt:
        logger.info("系统安全关闭")
    except Exception as e:
        logger.error(f"系统异常: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
