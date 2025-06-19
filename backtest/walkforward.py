# backtest/walkforward.py

import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
from sklearn.model_selection import ParameterGrid
from backtest.performance import calculate_performance_metrics

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('walkforward')

class WalkForwardBacktester:
    def __init__(self, data: pd.DataFrame, initial_train_size: float = 0.6, 
                 test_size: float = 0.2, n_splits: int = 5, 
                 fixed_train_size: bool = False):
        """
        初始化步进式回测器
        
        参数:
            data: 包含时间戳索引的DataFrame
            initial_train_size: 初始训练集占总数据的比例
            test_size: 每个测试集占总数据的比例
            n_splits: 分割次数
            fixed_train_size: 是否保持训练集大小固定(滚动窗口)
        """
        self.data = data.sort_index()
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.n_splits = n_splits
        self.fixed_train_size = fixed_train_size
        self._validate_parameters()
        
    def _validate_parameters(self):
        """验证参数有效性"""
        total_size = self.initial_train_size + self.test_size * self.n_splits
        if total_size > 1.0:
            raise ValueError(
                f"参数组合无效: initial_train_size + test_size * n_splits = {total_size} > 1.0")
    
    def _generate_splits(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """生成训练集和测试集分割"""
        splits = []
        n_samples = len(self.data)
        initial_train_end = int(n_samples * self.initial_train_size)
        
        # 初始训练集
        train_data = self.data.iloc[:initial_train_end]
        
        # 计算每个测试集的起点和终点
        test_size_samples = int(n_samples * self.test_size)
        
        for i in range(self.n_splits):
            test_start = initial_train_end + i * test_size_samples
            test_end = test_start + test_size_samples
            
            if test_end > n_samples:
                logger.warning(f"最后一次分割的测试集被截断: {test_end} > {n_samples}")
                test_end = n_samples
            
            test_data = self.data.iloc[test_start:test_end]
            
            splits.append((train_data.copy(), test_data.copy()))
            
            # 更新训练集: 滚动或扩展窗口
            if self.fixed_train_size:
                # 滚动窗口: 移除最早的部分数据，保持训练集大小不变
                train_data = train_data.iloc[test_size_samples:].append(test_data)
            else:
                # 扩展窗口: 保留所有历史数据
                train_data = train_data.append(test_data)
        
        return splits
    
    def run_backtest(self, strategy_class, strategy_params: Dict, 
                     metric: str = 'sharpe_ratio') -> pd.DataFrame:
        """
        运行步进式回测
        
        参数:
            strategy_class: 策略类
            strategy_params: 策略参数字典
            metric: 优化指标
            
        返回:
            包含每次分割结果的DataFrame
        """
        splits = self._generate_splits()
        results = []
        
        for i, (train_data, test_data) in enumerate(splits):
            logger.info(f"开始分割 {i+1}/{len(splits)}")
            logger.info(f"训练集: {train_data.index[0]} 到 {train_data.index[-1]} "
                       f"(共 {len(train_data)} 条)")
            logger.info(f"测试集: {test_data.index[0]} 到 {test_data.index[-1]} "
                       f"(共 {len(test_data)} 条)")
            
            # 在训练集上优化策略参数
            optimized_params = self._optimize_params(
                strategy_class, train_data, strategy_params, metric)
            
            # 在测试集上评估策略
            strategy = strategy_class(**optimized_params)
            strategy.run(test_data)
            
            # 计算性能指标
            returns = strategy.get_returns()
            metrics = calculate_performance_metrics(returns)
            
            # 保存结果
            result = {
                'split': i+1,
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'params': optimized_params,
                **metrics
            }
            results.append(result)
            
            logger.info(f"分割 {i+1} 完成. 夏普比率: {metrics['sharpe_ratio']:.2f}")
        
        return pd.DataFrame(results)
    
    def _optimize_params(self, strategy_class, data: pd.DataFrame, 
                         param_grid: Dict, metric: str) -> Dict:
        """
        在给定数据上优化策略参数
        
        参数:
            strategy_class: 策略类
            data: 用于优化的数据
            param_grid: 参数网格
            metric: 优化指标
            
        返回:
            最佳参数字典
        """
        best_metric = -np.inf if metric != 'max_drawdown' else np.inf
        best_params = None
        
        # 遍历所有参数组合
        for params in ParameterGrid(param_grid):
            try:
                strategy = strategy_class(**params)
                strategy.run(data)
                returns = strategy.get_returns()
                metrics = calculate_performance_metrics(returns)
                
                current_metric = metrics[metric]
                
                # 检查是否找到了更好的参数
                if ((metric != 'max_drawdown' and current_metric > best_metric) or 
                    (metric == 'max_drawdown' and current_metric < best_metric)):
                    best_metric = current_metric
                    best_params = params
                    
            except Exception as e:
                logger.warning(f"参数组合 {params} 失败: {str(e)}")
                continue
        
        if best_params is None:
            raise RuntimeError("没有找到有效的参数组合")
            
        logger.info(f"找到最佳参数: {best_params} ({metric}: {best_metric:.2f})")
        return best_params

if __name__ == "__main__":
    # 示例用法
    parser = argparse.ArgumentParser(description='步进式回测')
    parser.add_argument('--data', required=True, help='数据文件路径')
    parser.add_argument('--initial_train', type=float, default=0.6, 
                       help='初始训练集比例')
    parser.add_argument('--test_size', type=float, default=0.1, 
                       help='每个测试集比例')
    parser.add_argument('--n_splits', type=int, default=5, 
                       help='分割次数')
    parser.add_argument('--fixed_train', action='store_true',
                       help='使用固定大小的训练集(滚动窗口)')
    args = parser.parse_args()
    
    # 加载数据 (示例)
    data = pd.read_csv(args.data, index_col='date', parse_dates=True)
    
    # 定义策略参数网格 (示例)
    param_grid = {
        'window': [10, 20, 50],
        'threshold': [0.5, 1.0, 1.5],
    }
    
    # 运行步进式回测
    wf = WalkForwardBacktester(
        data, 
        initial_train_size=args.initial_train,
        test_size=args.test_size,
        n_splits=args.n_splits,
        fixed_train_size=args.fixed_train
    )
    
    # 这里需要替换为实际的策略类
    from strategies import MovingAverageCrossStrategy
    results = wf.run_backtest(MovingAverageCrossStrategy, param_grid)
    
    # 保存结果
    results.to_csv('walkforward_results.csv', index=False)
    print("步进式回测完成. 结果已保存到 walkforward_results.csv")
