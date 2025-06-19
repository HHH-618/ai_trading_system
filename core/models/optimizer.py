# core/models/optimizer.py

import numpy as np
import optuna
import tensorflow as tf
from typing import Dict, Any, Optional
from functools import partial
from configs.hyperparams import DQN_PARAMS, PPO_PARAMS
from utils.logger import setup_logger

logger = setup_logger('hyperparam_optimizer')

class HyperparameterOptimizer:
    """
    超参数优化系统，支持：
    1. 贝叶斯优化 (Optuna)
    2. 网格搜索
    3. 随机搜索
    4. 遗传算法
    """
    
    def __init__(self, model_type: str):
        self.model_type = model_type.lower()
        self.study = None
        
    def optimize_with_optuna(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 50,
        direction: str = 'maximize',
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        使用Optuna进行贝叶斯优化
        :param direction: 优化方向 (maximize/minimize)
        :param timeout: 最大优化时间(秒)
        :return: 最佳超参数组合
        """
        study = optuna.create_study(direction=direction)
        
        if self.model_type == 'dqn':
            objective = partial(self._dqn_objective, X=X, y=y)
        elif self.model_type == 'ppo':
            objective = partial(self._ppo_objective, X=X, y=y)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        self.study = study
        
        logger.info(f"最佳超参数: {study.best_params}")
        logger.info(f"最佳值: {study.best_value}")
        
        return study.best_params
    
    def _dqn_objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """DQN模型的优化目标函数"""
        params = {
            'gamma': trial.suggest_float('gamma', 0.9, 0.999),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'memory_size': trial.suggest_int('memory_size', 1000, 10000, step=1000),
            'target_update': trial.suggest_int('target_update', 50, 200),
            'epsilon_decay': trial.suggest_float('epsilon_decay', 0.99, 0.9999)
        }
        
        # 使用这些参数训练模型
        model = self._build_dqn_model(params)
        history = model.fit(X, y, verbose=0)
        
        # 返回需要优化的指标 (如验证集准确率)
        return max(history.history['val_accuracy'])
    
    def _ppo_objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """PPO模型的优化目标函数"""
        params = {
            'gamma': trial.suggest_float('gamma', 0.9, 0.999),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'clip_param': trial.suggest_float('clip_param', 0.1, 0.3),
            'entropy_coef': trial.suggest_float('entropy_coef', 0.001, 0.1),
            'epochs': trial.suggest_int('epochs', 3, 10)
        }
        
        # 使用这些参数训练模型
        model = self._build_ppo_model(params)
        history = model.fit(X, y, verbose=0)
        
        return max(history.history['val_accuracy'])
    
    def _build_dqn_model(self, params: Dict[str, Any]) -> tf.keras.Model:
        """根据参数构建DQN模型"""
        from core.strategies.rl.dqn import DQNStrategy
        return DQNStrategy(
            state_shape=X.shape[1:],
            action_size=3,  # buy, sell, hold
            **params
        )
    
    def _build_ppo_model(self, params: Dict[str, Any]) -> tf.keras.Model:
        """根据参数构建PPO模型"""
        from core.strategies.rl.ppo import PPOStrategy
        return PPOStrategy(
            input_dim=X.shape[1],
            action_dim=3,  # buy, sell, hold
            **params
        )
    
    def visualize_optimization(self):
        """可视化优化过程"""
        if self.study is None:
            logger.warning("没有优化研究可可视化")
            return
            
        try:
            import optuna.visualization as vis
            
            # 参数重要性图
            fig = vis.plot_param_importances(self.study)
            fig.show()
            
            # 优化过程图
            fig = vis.plot_optimization_history(self.study)
            fig.show()
            
        except ImportError:
            logger.warning("需要安装optuna可视化依赖: pip install optuna[visualization]")
    
    def grid_search(self, param_grid: Dict[str, List[Any]], 
                   X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """网格搜索超参数优化"""
        from sklearn.model_selection import ParameterGrid
        
        best_score = -np.inf
        best_params = None
        
        for params in ParameterGrid(param_grid):
            if self.model_type == 'dqn':
                model = self._build_dqn_model(params)
            else:
                model = self._build_ppo_model(params)
                
            history = model.fit(X, y, verbose=0)
            score = max(history.history['val_accuracy'])
            
            if score > best_score:
                best_score = score
                best_params = params
                
        logger.info(f"网格搜索最佳参数: {best_params}, 得分: {best_score}")
        return best_params
