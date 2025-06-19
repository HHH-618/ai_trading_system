# configs/constants.py

"""
系统全局常量定义
"""

# 时间相关常量
class TimeConstants:
    TRADING_DAY_HOURS = (0, 23)  # 交易时段(UTC)
    LONDON_OPEN = 8               # 伦敦开盘时间(UTC)
    NY_OPEN = 13                  # 纽约开盘时间(UTC)
    MARKET_CLOSE = 21             # 主要市场收盘时间(UTC)
    CANDLE_TIMEFRAMES = ['M1', 'M5', 'M15', 'H1', 'H4', 'D1']

# 订单相关常量
class OrderConstants:
    ORDER_TYPES = ['MARKET', 'LIMIT', 'STOP']
    ORDER_SIDES = ['BUY', 'SELL']
    ORDER_STATUS = ['OPEN', 'FILLED', 'PARTIALLY_FILLED', 'CANCELLED']
    SLIPPAGE_TOLERANCE = 0.0005  # 0.05%

# 风险相关常量
class RiskConstants:
    MAX_POSITION_SIZE = 0.1       # 最大仓位比例(10%)
    STOP_LOSS_PCT = 0.02          # 默认止损比例(2%)
    TAKE_PROFIT_PCT = 0.04        # 默认止盈比例(4%)
    RISK_FREE_RATE = 0.01         # 无风险利率(年化1%)

# 技术指标常量
class IndicatorConstants:
    EMA_PERIODS = [10, 20, 50, 100]
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9

# 模型相关常量
class ModelConstants:
    SEQ_LENGTH = 60               # 序列长度
    BATCH_SIZE = 64               # 训练批次大小
    VALIDATION_SPLIT = 0.2        # 验证集比例
