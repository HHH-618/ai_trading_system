# backtest/optimizer.py

import logging
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import ParameterGrid, ParameterSampler
from scipy.optimize import differential_evolution, minimize
from backtest.performance import calculate_performance_metrics

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('optimizer')

class StrategyOptimizer:
    def __init__(self, data: pd.DataFrame, strategy_class: Callable,
                 n_jobs: int = -1, random_state: int = None):
        """
        初始化策略优化器
        
        参数:
            data: 用于优化的数据
            strategy_class: 策略类
            n_jobs: 并行工作数(-1表示使用所有CPU核心)
            random_state: 随机种子
        """
        self.data = data
        self.strategy_class = strategy_class
        self.n_jobs = n_jobs if n_jobs > 0 else None
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def grid_search(self, param_grid: Dict, metric: str = 'sharpe_ratio') -> pd.DataFrame:
        """
        网格搜索参数优化
        
        参数:
            param_grid: 参数网格字典
            metric: 优化指标
            
        返回:
            包含所有参数组合结果的DataFrame
        """
        results = []
        param_combinations = list(ParameterGrid(param_grid))
        total_combinations = len(param_combinations)
        
        logger.info(f"开始网格搜索，共 {total_combinations} 种参数组合...")
        
        if self.n_jobs and self.n_jobs > 1 and total_combinations > 1:
            results = self._parallel_optimize(param_combinations, metric)
        else:
            for i, params in enumerate(param_combinations, 1):
                try:
                    result = self._evaluate_params(params, metric)
                    results.append(result)
                    logger.info(f"进度: {i}/{total_combinations} - {metric}: {result[metric]:.2f}")
                except Exception as e:
                    logger.warning(f"参数组合 {params} 失败: {str(e)}")
        
        df = pd.DataFrame(results)
        return df.sort_values(metric, ascending=metric == 'max_drawdown')
    
    def random_search(self, param_distributions: Dict, n_iter: int = 100,
                      metric: str = 'sharpe_ratio') -> pd.DataFrame:
        """
        随机搜索参数优化
        
        参数:
            param_distributions: 参数分布字典
            n_iter: 迭代次数
            metric: 优化指标
            
        返回:
            包含所有参数组合结果的DataFrame
        """
        results = []
        param_combinations = list(ParameterSampler(
            param_distributions, n_iter, random_state=self.random_state))
        
        logger.info(f"开始随机搜索，共 {n_iter} 次迭代...")
        
        if self.n_jobs and self.n_jobs > 1 and n_iter > 1:
            results = self._parallel_optimize(param_combinations, metric)
        else:
            for i, params in enumerate(param_combinations, 1):
                try:
                    result = self._evaluate_params(params, metric)
                    results.append(result)
                    logger.info(f"进度: {i}/{n_iter} - {metric}: {result[metric]:.2f}")
                except Exception as e:
                    logger.warning(f"参数组合 {params} 失败: {str(e)}")
        
        df = pd.DataFrame(results)
        return df.sort_values(metric, ascending=metric == 'max_drawdown')
    
    def bayesian_optimization(self, param_bounds: Dict, n_iter: int = 50,
                             metric: str = 'sharpe_ratio') -> pd.DataFrame:
        """
        贝叶斯优化(使用差分进化作为替代)
        
        参数:
            param_bounds: 参数边界字典 {param_name: (min, max)}
            n_iter: 迭代次数
            metric: 优化指标
            
        返回:
            包含优化结果的DataFrame
        """
        logger.info(f"开始差分进化优化，共 {n_iter} 次迭代...")
        
        # 将参数名和边界转换为差分进化需要的格式
        param_names = list(param_bounds.keys())
        bounds = [param_bounds[name] for name in param_names]
        
        def objective_function(x):
            # 将数值向量转换回参数字典
            params = {name: x[i] for i, name in enumerate(param_names)}
            
            try:
                result = self._evaluate_params(params, metric)
                # 对于最大回撤，我们需要最小化
                score = -result[metric] if metric != 'max_drawdown' else result[metric]
                return score
            except Exception as e:
                logger.warning(f"参数评估失败: {str(e)}")
                return np.inf if metric != 'max_drawdown' else -np.inf
        
        # 运行差分进化优化
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=n_iter,
            popsize=15,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=self.random_state,
            workers=self.n_jobs,
            disp=True
        )
        
        # 准备结果
        best_params = {name: result.x[i] for i, name in enumerate(param_names)}
        best_metric = -result.fun if metric != 'max_drawdown' else result.fun
        
        logger.info(f"优化完成. 最佳 {metric}: {best_metric:.2f}")
        logger.info(f"最佳参数: {best_params}")
        
        return pd.DataFrame([{'params': best_params, metric: best_metric}])
    
    def _parallel_optimize(self, param_combinations: List[Dict], 
                           metric: str) -> List[Dict]:
        """并行执行参数评估"""
        results = []
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {
                executor.submit(self._evaluate_params, params, metric): params
                for params in param_combinations
            }
            
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"进度: {i}/{len(param_combinations)} - {metric}: {result[metric]:.2f}")
                except Exception as e:
                    params = futures[future]
                    logger.warning(f"参数组合 {params} 失败: {str(e)}")
        
        return results
    
    def _evaluate_params(self, params: Dict, metric: str) -> Dict:
        """评估单个参数组合"""
        strategy = self.strategy_class(**params)
        strategy.run(self.data)
        returns = strategy.get_returns()
        metrics = calculate_performance_metrics(returns)
        
        return {
            'params': params,
            **metrics
        }
    
    def analyze_parameter_space(self, results_df: pd.DataFrame, 
                                target_metric: str = 'sharpe_ratio') -> Dict:
        """
        分析参数空间
        
        参数:
            results_df: 包含优化结果的DataFrame
            target_metric: 目标指标
            
        返回:
            包含参数分析结果的字典
        """
        analysis = {}
        
        # 确保结果中包含参数
        if 'params' not in results_df.columns:
            raise ValueError("结果DataFrame必须包含'params'列")
        
        # 提取最佳参数组合
        ascending = target_metric == 'max_drawdown'
        best_result = results_df.sort_values(target_metric, ascending=ascending).iloc[0]
        analysis['best_params'] = best_result['params']
        analysis['best_metric'] = best_result[target_metric]
        
        # 参数重要性分析
        param_importance = {}
        param_names = list(best_result['params'].keys())
        
        for param in param_names:
            # 计算参数值与目标指标的相关性
            param_values = results_df['params'].apply(lambda x: x[param])
            correlation = np.corrcoef(param_values, results_df[target_metric])[0, 1]
            param_importance[param] = correlation
        
        analysis['param_importance'] = dict(
            sorted(param_importance.items(), key=lambda x: abs(x[1]), reverse=True))
        
        # 过拟合检测 (通过比较训练集和验证集性能)
        if 'train_metric' in results_df.columns and 'test_metric' in results_df.columns:
            overfitting_score = (
                results_df['train_metric'].mean() - results_df['test_metric'].mean())
            analysis['overfitting_risk'] = overfitting_score
        
        return analysis

if __name__ == "__main__":
    # 示例用法
    parser = argparse.ArgumentParser(description='策略参数优化')
    parser.add_argument('--data', required=True, help='数据文件路径')
    parser.add_argument('--method', choices=['grid', 'random', 'bayesian'], 
                       default='grid', help='优化方法')
    parser.add_argument('--metric', default='sharpe_ratio', 
                       help='优化指标 (sharpe_ratio, sortino_ratio, etc.)')
    parser.add_argument('--n_jobs', type=int, default=-1, 
                       help='并行工作数 (-1表示使用所有核心)')
    parser.add_argument('--output', default='optimization_results.csv', 
                       help='输出文件路径')
    args = parser.parse_args()
    
    # 加载数据 (示例)
    data = pd.read_csv(args.data, index_col='date', parse_dates=True)
    
    # 定义策略参数 (示例)
    from strategies import MovingAverageCrossStrategy
    
    if args.method == 'grid':
        param_grid = {
            'window_fast': [5, 10, 20],
            'window_slow': [20, 50, 100],
            'threshold': [0.005, 0.01, 0.02],
        }
    else:
        param_distributions = {
            'window_fast': np.arange(5, 50),
            'window_slow': np.arange(20, 200),
            'threshold': np.linspace(0.001, 0.05, 100),
        }
    
    # 创建优化器
    optimizer = StrategyOptimizer(
        data, 
        MovingAverageCrossStrategy,
        n_jobs=args.n_jobs,
        random_state=42
    )
    
    # 运行优化
    if args.method == 'grid':
        results = optimizer.grid_search(param_grid, args.metric)
    elif args.method == 'random':
        results = optimizer.random_search(param_distributions, 100, args.metric)
    else:
        param_bounds = {
            'window_fast': (5, 50),
            'window_slow': (20, 200),
            'threshold': (0.001, 0.05),
        }
        results = optimizer.bayesian_optimization(param_bounds, 50, args.metric)
    
    # 分析结果
    analysis = optimizer.analyze_parameter_space(results, args.metric)
    print("\n参数分析结果:")
    print(f"最佳 {args.metric}: {analysis['best_metric']:.2f}")
    print("最佳参数:", analysis['best_params'])
    print("\n参数重要性:")
    for param, importance in analysis['param_importance'].items():
        print(f"{param}: {importance:.2f}")
    
    # 保存结果
    results.to_csv(args.output, index=False)
    print(f"\n优化结果已保存到 {args.output}")
