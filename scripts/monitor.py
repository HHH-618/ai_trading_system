# scripts/monitor.py

import time
import logging
import argparse
import psutil
import requests
import smtplib
from datetime import datetime
from prometheus_client import start_http_server, Gauge, Counter
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('monitor')

# Prometheus指标
CPU_USAGE = Gauge('ai_trader_cpu_usage', 'CPU使用率百分比')
MEMORY_USAGE = Gauge('ai_trader_memory_usage', '内存使用率百分比')
LATENCY = Gauge('ai_trader_latency', 'API响应延迟(ms)')
ERROR_COUNT = Counter('ai_trader_errors', '错误计数')
TRADES_COUNT = Counter('ai_trader_trades', '交易计数')

class TradingSystemMonitor:
    def __init__(self, config_path='config/monitor_config.yaml'):
        self.config = self._load_config(config_path)
        self.slack_client = WebClient(token=self.config['slack']['token']) if self.config.get('slack') else None
        self.check_interval = self.config.get('check_interval', 60)
        
    def _load_config(self, config_path):
        """加载监控配置文件"""
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    def _check_system_resources(self):
        """检查系统资源使用情况"""
        cpu_percent = psutil.cpu_percent(interval=1)
        mem_percent = psutil.virtual_memory().percent
        
        CPU_USAGE.set(cpu_percent)
        MEMORY_USAGE.set(mem_percent)
        
        if cpu_percent > self.config.get('cpu_threshold', 90):
            self._trigger_alert(f"高CPU使用率: {cpu_percent}%")
        if mem_percent > self.config.get('memory_threshold', 90):
            self._trigger_alert(f"高内存使用率: {mem_percent}%")
    
    def _check_api_health(self):
        """检查交易API健康状况"""
        endpoints = self.config.get('api_endpoints', [])
        for endpoint in endpoints:
            try:
                start_time = time.time()
                response = requests.get(
                    endpoint['url'],
                    timeout=endpoint.get('timeout', 5),
                    headers=endpoint.get('headers', {})
                )
                latency = (time.time() - start_time) * 1000  # ms
                LATENCY.set(latency)
                
                if not response.ok:
                    ERROR_COUNT.inc()
                    self._trigger_alert(f"API {endpoint['name']} 响应异常: {response.status_code}")
                
                if latency > endpoint.get('latency_threshold', 500):
                    self._trigger_alert(f"API {endpoint['name']} 高延迟: {latency:.2f}ms")
                    
            except Exception as e:
                ERROR_COUNT.inc()
                self._trigger_alert(f"API {endpoint['name']} 检查失败: {str(e)}")
    
    def _check_trading_activity(self):
        """检查交易活动"""
        # 这里可以连接到数据库或API获取交易数据
        # 示例: 检查最近是否有交易
        try:
            # 模拟获取交易数量
            recent_trades = 5  # 实际应从数据库获取
            TRADES_COUNT.inc(recent_trades)
            
            if recent_trades == 0 and self.config.get('alert_on_no_trades'):
                self._trigger_alert("警告: 最近没有检测到交易活动")
        except Exception as e:
            logger.error(f"检查交易活动失败: {e}")
            ERROR_COUNT.inc()
    
    def _trigger_alert(self, message):
        """触发警报"""
        logger.warning(message)
        
        # 发送Slack通知
        if self.slack_client and self.config['slack'].get('enabled'):
            try:
                self.slack_client.chat_postMessage(
                    channel=self.config['slack']['channel'],
                    text=f"[AI交易系统警报] {message}"
                )
            except SlackApiError as e:
                logger.error(f"Slack通知发送失败: {e.response['error']}")
        
        # 发送邮件通知
        if self.config.get('email_alerts', {}).get('enabled'):
            self._send_email_alert(message)
    
    def _send_email_alert(self, message):
        """发送邮件警报"""
        try:
            with smtplib.SMTP(
                self.config['email_alerts']['smtp_server'],
                self.config['email_alerts']['smtp_port']
            ) as server:
                server.starttls()
                server.login(
                    self.config['email_alerts']['username'],
                    self.config['email_alerts']['password']
                )
                
                subject = "AI交易系统警报"
                body = f"""
                时间: {datetime.now()}
                系统: AI交易系统
                警报内容: {message}
                """
                
                msg = f"Subject: {subject}\n\n{body}"
                server.sendmail(
                    self.config['email_alerts']['from'],
                    self.config['email_alerts']['to'],
                    msg
                )
        except Exception as e:
            logger.error(f"邮件警报发送失败: {e}")
    
    def _auto_recover(self):
        """尝试自动恢复"""
        # 这里可以实现自动恢复逻辑，比如重启服务等
        if self.config.get('auto_recovery', {}).get('enabled'):
            logger.info("尝试自动恢复...")
            # 示例: 重启Docker容器
            try:
                subprocess.run(
                    ["docker", "restart", "ai-trader"],
                    check=True,
                    timeout=30
                )
                logger.info("自动恢复成功: 交易容器已重启")
            except subprocess.SubprocessError as e:
                logger.error(f"自动恢复失败: {e}")
                self._trigger_alert("自动恢复失败，需要人工干预")
    
    def run(self):
        """运行监控循环"""
        logger.info("启动AI交易系统监控...")
        
        # 启动Prometheus指标服务器
        start_http_server(self.config.get('prometheus_port', 8000))
        
        while True:
            try:
                logger.info("执行监控检查...")
                self._check_system_resources()
                self._check_api_health()
                self._check_trading_activity()
                
                time.sleep(self.check_interval)
            except KeyboardInterrupt:
                logger.info("监控服务停止")
                break
            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                self._trigger_alert(f"监控系统出错: {e}")
                time.sleep(self.check_interval)

class EnhancedTradingMonitor(TradingSystemMonitor):
    def __init__(self, config_path):
        super().__init__(config_path)
        # 新增监控指标
        self.latency_metrics = Gauge('trade_latency', '交易执行延迟', ['strategy'])
        self.slippage_metrics = Gauge('trade_slippage', '交易滑点', ['symbol'])
        
    def _check_trading_performance(self):
        """增强版交易绩效监控"""
        # 1. 从数据库获取最近100笔交易
        trades = self._query_recent_trades(limit=100)
    
        if not trades:
            self._trigger_alert("未检测到近期交易活动")
            return
        
        # 2. 计算关键指标
        win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades)
        avg_slippage = np.mean([t['slippage'] for t in trades])
        sharpe_ratio = self._calculate_sharpe_ratio([t['pnl'] for t in trades])
    
        # 3. 设置阈值告警
        if win_rate < 0.4:
            self._trigger_alert(f"胜率下降至{win_rate:.1%}")
        if avg_slippage > 0.0005:  # 5 pips
            self._trigger_alert(f"平均滑点过高: {avg_slippage:.4f}")
        if sharpe_ratio < 1.0:
            self._trigger_alert(f"夏普比率下降至{sharpe_ratio:.2f}")
        
        # 4. 记录指标
        self.win_rate_metric.set(win_rate)
        self.slippage_metrics.labels(trades[0]['symbol']).set(avg_slippage)
        self.sharpe_metric.set(sharpe_ratio)

    def _query_recent_trades(self, limit: int = 100) -> List[Dict]:
        """从数据库查询最近交易"""
        # 这里应该是实际的数据库查询代码
        # 模拟返回数据
        return [
            {
                'symbol': 'EURUSD',
                'entry_price': 1.0800,
                'exit_price': 1.0820,
                'pnl': 200,
                'slippage': 0.0001,
                'timestamp': '2023-01-01 12:00:00'
            }
            # ...更多交易记录
        ]

    def _calculate_sharpe_ratio(self, pnls: List[float]) -> float:
        """计算夏普比率"""
        if not pnls or len(pnls) < 2:
            return 0.0
    
        returns = np.array(pnls) / np.mean(np.abs(pnls))  # 标准化处理
        excess_returns = returns - 0.01/252  # 假设无风险利率1%年化
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def _auto_recovery(self):
        """智能自愈机制"""
        error_count = self._get_error_count(last_minutes=5)
        if error_count > 10:
            logger.critical("触发自动恢复流程")
            try:
                self._restart_services()
                self._rollback_strategy()
                self._notify_ops("自动恢复已执行")
            except Exception as e:
                self._trigger_alert(f"自动恢复失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AI交易系统监控脚本')
    parser.add_argument('--config', default='config/monitor_config.yaml',
                       help='监控配置文件路径')
    args = parser.parse_args()
    
    monitor = TradingSystemMonitor(args.config)
    monitor.run()
