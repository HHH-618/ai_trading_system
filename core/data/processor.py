# core/data/processor.py

import pandas as pd
import numpy as np
import talib
from typing import Optional, Dict
from configs.constants import IndicatorConstants
from utils.logger import setup_logger
from utils.helpers import safe_divide
from configs.constants import TimeConstants

logger = setup_logger('data_processor')

class DataProcessor:
    """高级金融数据处理与特征工程"""
    
    def __init__(self):
        self.scalers = {}
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        计算技术指标
        :param df: 包含OHLCV数据的DataFrame
        :return: 包含技术指标的DataFrame
        """
        if df.empty:
            return None
            
        try:
            df = df.copy()
            closes = df['close'].values.astype(np.float64)
            highs = df['high'].values.astype(np.float64)
            lows = df['low'].values.astype(np.float64)
            volumes = df['tick_volume'].values.astype(np.float64)
            
            # 趋势指标
            for period in IndicatorConstants.EMA_PERIODS:
                df[f'EMA{period}'] = talib.EMA(closes, timeperiod=period)
                
            # 震荡指标
            df['RSI'] = talib.RSI(closes, timeperiod=IndicatorConstants.RSI_PERIOD)
            df['ADX'] = talib.ADX(highs, lows, closes, timeperiod=14)
            df['CCI'] = talib.CCI(highs, lows, closes, timeperiod=20)
            
            # 动量指标
            df['MACD'], df['MACD_Signal'], _ = talib.MACD(
                closes, 
                fastperiod=IndicatorConstants.MACD_FAST,
                slowperiod=IndicatorConstants.MACD_SLOW,
                signalperiod=IndicatorConstants.MACD_SIGNAL
            )
            
            # 波动率指标
            df['ATR'] = talib.ATR(
                highs, lows, closes, 
                timeperiod=IndicatorConstants.ATR_PERIOD
            )
            df['NATR'] = talib.NATR(highs, lows, closes, timeperiod=14)
            
            # 成交量指标
            df['OBV'] = talib.OBV(closes, volumes)
            
            # 自定义特征
            df = self._create_custom_features(df)
            
            # 处理缺失值
            df.ffill(inplace=True)
            df.bfill(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"计算技术指标失败: {str(e)}")
            return None
            
    def _create_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建自定义特征"""
        # 价格变化特征
        df['price_change'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # 波动率特征
        df['volatility_5'] = df['price_change'].rolling(5).std()
        df['volatility_20'] = df['price_change'].rolling(20).std()
        
        # 量价关系
        df['volume_spike'] = (df['tick_volume'] / df['tick_volume'].rolling(20).mean() - 1)
        df['price_volume_corr'] = df['close'].rolling(20).corr(df['tick_volume'])
        
        # 时间特征
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_london_open'] = ((df['hour'] >= TimeConstants.LONDON_OPEN) & 
                               (df['hour'] < TimeConstants.NY_OPEN)).astype(int)
        df['is_ny_open'] = ((df['hour'] >= TimeConstants.NY_OPEN) & 
                           (df['hour'] < TimeConstants.MARKET_CLOSE)).astype(int)
        
        return df
        
    def normalize_data(self, df: pd.DataFrame, method: str = 'robust') -> Optional[pd.DataFrame]:
        """
        数据标准化
        :param df: 输入数据
        :param method: 标准化方法 (robust/minmax/zscore)
        """
        from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

        if df is None:
            logger.error("输入数据为None，请检查数据处理流程")
            raise ValueError("输入数据不能为None")
        
        if df.empty:
            logger.warning("输入数据为空DataFrame")
            return None
            
        try:
            scalers = {
                'robust': RobustScaler(),
                'minmax': MinMaxScaler(),
                'zscore': StandardScaler()
            }
            
            if method not in scalers:
                logger.warning(f"未知的标准化方法: {method}, 使用robust")
                method = 'robust'
                
            # 对每组特征分别标准化
            feature_groups = {
                'price': ['open', 'high', 'low', 'close'],
                'volume': ['tick_volume', 'OBV'],
                'indicators': ['EMA10', 'EMA20', 'RSI', 'MACD']
            }
            
            scaled_features = []
            for group, cols in feature_groups.items():
                valid_cols = [c for c in cols if c in df.columns]
                if not valid_cols:
                    continue
                    
                scaler = scalers[method]
                scaled = scaler.fit_transform(df[valid_cols])
                self.scalers[group] = scaler
                scaled_features.append(scaled)
                
            # 合并所有特征
            if scaled_features:
                df_scaled = pd.DataFrame(
                    np.concatenate(scaled_features, axis=1),
                    index=df.index,
                    columns=[f"{col}_scaled" for group in feature_groups.values() for col in group if col in df.columns]
                )
                return pd.concat([df, df_scaled], axis=1)
            return df
            
        except Exception as e:
            logger.error(f"数据标准化失败: {str(e)}")
            return None

    class DataProcessor:
        def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
            """统一的数据处理方法（示例）"""
            processed = self.calculate_technical_indicators(data)
            normalized = self.normalize_data(processed)
            return normalized.dropna()

# 单例模式
data_processor = DataProcessor()
