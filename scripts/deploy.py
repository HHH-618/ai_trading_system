# scripts/deploy.py

import os
import logging
import argparse
import subprocess
from datetime import datetime
import yaml
import docker
import boto3
from fabric import Connection

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('deploy')

class AITradingDeployer:
    def __init__(self, config_path='config/deploy_config.yaml'):
        self.config = self._load_config(config_path)
        self.docker_client = docker.from_env()
        self.aws_client = boto3.client('ecs') if self.config.get('aws_deploy') else None
        
    def _load_config(self, config_path):
        """加载部署配置文件"""
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    def _validate_environment(self):
        """验证部署环境"""
        logger.info("验证部署环境...")
        # 检查必要的环境变量
        required_vars = ['API_KEY', 'API_SECRET', 'DB_URL']
        for var in required_vars:
            if var not in os.environ:
                raise EnvironmentError(f"缺少必要环境变量: {var}")
    
    def _build_docker_image(self):
        """构建Docker镜像"""
        logger.info("构建Docker交易镜像...")
        try:
            image, build_logs = self.docker_client.images.build(
                path=".",
                tag=f"ai-trader:{datetime.now().strftime('%Y%m%d%H%M')}",
                dockerfile="Dockerfile.prod"
            )
            for chunk in build_logs:
                if 'stream' in chunk:
                    logger.debug(chunk['stream'].strip())
            return image
        except docker.errors.BuildError as e:
            logger.error(f"Docker构建失败: {e}")
            raise
    
    def _deploy_to_aws(self, image):
        """部署到AWS ECS"""
        if not self.aws_client:
            return
            
        logger.info("部署到AWS ECS...")
        # 更新ECS任务定义
        response = self.aws_client.register_task_definition(
            family='ai-trader',
            containerDefinitions=[
                {
                    'name': 'ai-trader',
                    'image': image.tags[0],
                    'cpu': self.config['aws'].get('cpu', 1024),
                    'memory': self.config['aws'].get('memory', 2048),
                    'essential': True,
                    'environment': [
                        {'name': 'API_KEY', 'value': os.getenv('API_KEY')},
                        {'name': 'API_SECRET', 'value': os.getenv('API_SECRET')},
                        {'name': 'DB_URL', 'value': os.getenv('DB_URL')},
                    ],
                }
            ],
            networkMode='awsvpc',
            requiresCompatibilities=['FARGATE'],
        )
        task_arn = response['taskDefinition']['taskDefinitionArn']
        
        # 更新服务
        self.aws_client.update_service(
            cluster=self.config['aws']['cluster'],
            service=self.config['aws']['service'],
            taskDefinition=task_arn,
        )
        logger.info(f"服务已更新，使用任务定义: {task_arn}")
    
    def _deploy_to_bare_metal(self, image):
        """部署到物理服务器"""
        if not self.config.get('bare_metal'):
            return
            
        logger.info("部署到物理服务器...")
        for host in self.config['bare_metal']['hosts']:
            conn = Connection(
                host=host['ip'],
                user=host['user'],
                connect_kwargs={"key_filename": host.get('ssh_key')}
            )
            
            # 停止现有容器
            conn.run('docker stop ai-trader || true')
            conn.run('docker rm ai-trader || true')
            
            # 拉取新镜像
            conn.run(f'docker pull {image.tags[0]}')
            
            # 启动新容器
            cmd = f"""
            docker run -d \
                --name ai-trader \
                --restart unless-stopped \
                -e API_KEY={os.getenv('API_KEY')} \
                -e API_SECRET={os.getenv('API_SECRET')} \
                -e DB_URL={os.getenv('DB_URL')} \
                -v /var/log/ai-trader:/app/logs \
                {image.tags[0]}
            """
            conn.run(cmd)
            logger.info(f"{host['ip']} 部署完成")
    
    def run_health_check(self):
        """运行健康检查"""
        logger.info("运行部署后健康检查...")
        # 这里可以添加更复杂的健康检查逻辑
        try:
            subprocess.run(["curl", "-f", "http://localhost:8080/health"], check=True)
            logger.info("健康检查通过")
            return True
        except subprocess.CalledProcessError:
            logger.error("健康检查失败")
            return False
    
    def deploy(self):
        """执行完整部署流程"""
        try:
            self._validate_environment()
            image = self._build_docker_image()
            
            if self.config.get('aws_deploy'):
                self._deploy_to_aws(image)
            
            if self.config.get('bare_metal'):
                self._deploy_to_bare_metal(image)
            
            if not self.run_health_check():
                raise RuntimeError("部署后健康检查失败")
                
            logger.info("✅ 部署成功完成")
        except Exception as e:
            logger.error(f"部署失败: {e}")
            raise

class EnhancedDeployer(AITradingDeployer):
    def _health_check(self):
        """增强版健康检查"""
        checks = {
            'api_connect': self._check_api_connectivity(),
            'model_loading': self._check_model_loading(),
            'data_pipeline': self._check_data_pipeline()
        }
        
        if not all(checks.values()):
            failed = [k for k, v in checks.items() if not v]
            raise RuntimeError(f"健康检查失败: {failed}")
            
    def _check_model_loading(self):
        try:
            test_data = np.random.rand(1, 60, 20)
            for model in self.model_ensemble.models:
                model.predict(test_data)
            return True
        except Exception as e:
            logger.error(f"模型加载测试失败: {e}")
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AI交易系统部署脚本')
    parser.add_argument('--config', default='config/deploy_config.yaml',
                       help='部署配置文件路径')
    args = parser.parse_args()
    
    deployer = AITradingDeployer(args.config)
    deployer.deploy()
