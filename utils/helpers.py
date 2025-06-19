# utils/helpers.py

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, List
from datetime import datetime, timedelta

def safe_divide(a: Union[np.ndarray, pd.Series, float], 
                b: Union[np.ndarray, pd.Series, float]) -> Union[np.ndarray, pd.Series, float]:
    """
    安全的除法运算，避免除以零
    :return: a / b (当b=0时返回0)
    """
    if isinstance(a, (pd.Series, pd.DataFrame)) or isinstance(b, (pd.Series, pd.DataFrame)):
        return a.div(b, fill_value=0)
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

def calculate_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """
    计算百分比变化，处理缺失值
    :param periods: 计算周期
    """
    return series.pct_change(periods).fillna(0)

def resample_data(df: pd.DataFrame, 
                 rule: str = '1H', 
                 agg: Dict[str, str] = None) -> pd.DataFrame:
    """
    重采样时间序列数据
    :param rule: 重采样规则 (e.g. '1H', '15T', '1D')
    :param agg: 聚合方法 {'column': 'method'}
    """
    if agg is None:
        agg = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
    
    # 只保留需要的列
    cols = [c for c in agg.keys() if c in df.columns]
    return df[cols].resample(rule).agg(agg)

def calculate_sharpe_ratio(returns: pd.Series, 
                          risk_free_rate: float = 0.01,
                          annualize: bool = True) -> float:
    """
    计算夏普比率
    :param returns: 收益率序列
    :param risk_free_rate: 无风险利率 (年化)
    :param annualize: 是否年化
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / 252)
    sharpe = excess_returns.mean() / excess_returns.std()
    
    if annualize:
        sharpe *= np.sqrt(252)
        
    return sharpe

def calculate_max_drawdown(equity: pd.Series) -> float:
    """
    计算最大回撤
    :param equity: 资金曲线
    :return: 最大回撤比例 (0-1)
    """
    peak = equity.cummax()
    drawdown = (peak - equity) / peak
    return drawdown.max()

def generate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    生成时间特征
    :return: 添加了时间特征的DataFrame
    """
    df = df.copy()
    index = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['time'])
    
    df['hour'] = index.hour
    df['day_of_week'] = index.dayofweek
    df['day_of_month'] = index.day
    df['month'] = index.month
    
    # 市场时段特征
    df['is_london_open'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['is_ny_open'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
    
    return df

def split_train_test(df: pd.DataFrame, 
                    test_size: float = 0.2, 
                    time_based: bool = True) -> tuple:
    """
    分割训练集和测试集
    :param test_size: 测试集比例
    :param time_based: 是否按时间分割 (True=后20%作为测试集)
    """
    if time_based:
        split_idx = int(len(df) * (1 - test_size))
        return df.iloc[:split_idx], df.iloc[split_idx:]
    else:
        from sklearn.model_selection import train_test_split
        return train_test_split(df, test_size=test_size, shuffle=False)
