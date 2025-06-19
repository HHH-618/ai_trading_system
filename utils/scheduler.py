# utils/scheduler.py

import time
import schedule
import threading
from typing import Callable, Optional
from datetime import datetime, time as dt_time
from utils.logger import setup_logger

logger = setup_logger('scheduler')

class TradingScheduler:
    """交易任务调度器"""
    
    def __init__(self):
        self.jobs = []
        self._stop_event = threading.Event()
        self._scheduler_thread = None
        
    def add_daily_job(self, 
                     task: Callable, 
                     time_str: str, 
                     args: Optional[tuple] = None,
                     kwargs: Optional[dict] = None):
        """
        添加每日定时任务
        :param task: 要执行的任务函数
        :param time_str: 时间字符串 "HH:MM"
        :param args: 位置参数
        :param kwargs: 关键字参数
        """
        hour, minute = map(int, time_str.split(':'))
        job = schedule.every().day.at(time_str).do(
            self._wrap_task, task, args or (), kwargs or {}
        )
        self.jobs.append(job)
        logger.info(f"添加每日任务: {task.__name__} @ {time_str}")
        
    def add_intraday_job(self, 
                        task: Callable, 
                        interval_minutes: int,
                        args: Optional[tuple] = None,
                        kwargs: Optional[dict] = None):
        """
        添加日内定时任务
        :param interval_minutes: 间隔分钟数
        """
        job = schedule.every(interval_minutes).minutes.do(
            self._wrap_task, task, args or (), kwargs or {}
        )
        self.jobs.append(job)
        logger.info(f"添加日内任务: {task.__name__} 每 {interval_minutes} 分钟")
        
    def _wrap_task(self, task: Callable, args: tuple, kwargs: dict):
        """包装任务函数，添加日志和异常处理"""
        logger.info(f"开始执行任务: {task.__name__}")
        try:
            start_time = time.time()
            result = task(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"任务完成: {task.__name__} (耗时: {elapsed:.2f}s)")
            return result
        except Exception as e:
            logger.error(f"任务失败: {task.__name__} - {str(e)}")
            raise
        
    def start(self):
        """启动调度器"""
        if self._scheduler_thread is not None:
            logger.warning("调度器已在运行")
            return
            
        self._stop_event.clear()
        self._scheduler_thread = threading.Thread(
            target=self._run_scheduler,
            daemon=True
        )
        self._scheduler_thread.start()
        logger.info("任务调度器已启动")
        
    def _run_scheduler(self):
        """调度器运行循环"""
        while not self._stop_event.is_set():
            schedule.run_pending()
            time.sleep(1)
            
    def stop(self):
        """停止调度器"""
        if self._scheduler_thread is None:
            logger.warning("调度器未运行")
            return
            
        self._stop_event.set()
        self._scheduler_thread.join()
        self._scheduler_thread = None
        schedule.clear()
        self.jobs = []
        logger.info("任务调度器已停止")
        
    def is_market_hours(self, 
                       open_time: str = "09:30", 
                       close_time: str = "16:00") -> bool:
        """
        检查当前是否在交易时段内
        :param open_time: 开盘时间 "HH:MM"
        :param close_time: 收盘时间 "HH:MM"
        """
        now = datetime.now().time()
        open_t = dt_time(*map(int, open_time.split(':')))
        close_t = dt_time(*map(int, close_time.split(':')))
        return open_t <= now <= close_t
