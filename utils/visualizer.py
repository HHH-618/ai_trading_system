# utils/visualizer.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List
from configs.settings import Config

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False    # 负号显示

class TradingVisualizer:
    """交易可视化工具"""
    
    @staticmethod
    def plot_equity_curve(equity: pd.Series, 
                         title: str = "资金曲线",
                         save_path: Optional[Path] = None):
        """
        绘制资金曲线
        :param equity: 资金序列 (index为时间)
        :param title: 图表标题
        :param save_path: 图片保存路径
        """
        plt.figure(figsize=(12, 6))
        
        # 计算回撤
        drawdown = (equity.cummax() - equity) / equity.cummax()
        max_drawdown = drawdown.max()
        max_drawdown_pos = drawdown.idxmax()
        
        # 主图: 资金曲线
        ax1 = plt.subplot(2, 1, 1)
        equity.plot(linewidth=2, color='royalblue')
        plt.title(f"{title} (最大回撤: {max_drawdown:.2%})")
        plt.ylabel("资金")
        plt.grid(True)
        
        # 标注最大回撤
        ax1.axvline(max_drawdown_pos, color='red', linestyle='--', alpha=0.7)
        ax1.text(max_drawdown_pos, equity.iloc[0], 
                f'最大回撤: {max_drawdown:.2%}',
                color='red', ha='right')
        
        # 副图: 回撤曲线
        plt.subplot(2, 1, 2, sharex=ax1)
        drawdown.plot(color='darkorange', linewidth=1.5)
        plt.fill_between(drawdown.index, drawdown, color='darkorange', alpha=0.3)
        plt.ylabel("回撤")
        plt.xlabel("时间")
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_trade_signals(prices: pd.Series, 
                          signals: Dict[str, List[pd.Timestamp]],
                          title: str = "交易信号",
                          save_path: Optional[Path] = None):
        """
        绘制价格和交易信号
        :param prices: 价格序列
        :param signals: 信号字典 {'buy': [timestamps], 'sell': [timestamps]}
        :param title: 图表标题
        """
        plt.figure(figsize=(12, 6))
        
        # 价格曲线
        prices.plot(linewidth=1.5, color='black', alpha=0.8, label='价格')
        
        # 买卖信号
        if 'buy' in signals and signals['buy']:
            buy_dates = [d for d in signals['buy'] if d in prices.index]
            plt.scatter(buy_dates, prices[buy_dates], 
                       color='green', marker='^', s=100, label='买入')
        
        if 'sell' in signals and signals['sell']:
            sell_dates = [d for d in signals['sell'] if d in prices.index]
            plt.scatter(sell_dates, prices[sell_dates], 
                       color='red', marker='v', s=100, label='卖出')
        
        plt.title(title)
        plt.ylabel("价格")
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_feature_importance(feature_importance: Dict[str, float],
                              title: str = "特征重要性",
                              save_path: Optional[Path] = None):
        """
        绘制特征重要性
        :param feature_importance: 特征重要性字典 {'feature': importance}
        :param title: 图表标题
        """
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
        
        # 排序
        sorted_idx = np.argsort(importances)
        features = [features[i] for i in sorted_idx]
        importances = [importances[i] for i in sorted_idx]
        
        plt.figure(figsize=(10, 6))
        plt.barh(features, importances, color='teal', alpha=0.7)
        plt.title(title)
        plt.xlabel("重要性得分")
        plt.grid(True, axis='x', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
