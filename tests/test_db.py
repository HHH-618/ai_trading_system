# tests/test_db.py

from storage import DataStorage

storage = DataStorage()
print("数据库连接成功！")

# 测试查询
import pandas as pd
from datetime import datetime

test_data = pd.DataFrame({
    'open': [1.08], 'high': [1.09], 
    'low': [1.07], 'close': [1.085],
    'volume': [1000]
}, index=[datetime.now()])

storage.save_candles('EURUSD', '1H', test_data)
print("数据插入成功！")
