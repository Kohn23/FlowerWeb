"""
    1、项目运行时的模型管理类，在项目启动时根据model_registry注册的模型类别进行模型加载
"""

import logging
import threading
from model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelManager:
    _loaded_models = {}
    _lock = threading.Lock()

    @classmethod
    def load_all_models(cls):
        with cls._lock:  # 确保线程安全
            for model_name in ModelRegistry._models:
                config = ModelRegistry.get_model_config(model_name)
                model_class = config["class"]
                model_path = config["model_path"]
                try:
                    model = model_class(model_path)
                    model.load_model()
                    cls._loaded_models[model_name] = model
                    logger.info(f"模型加载成功: {model_name}")
                except Exception as e:
                    logger.error(f"模型加载失败: {model_name}, 错误: {str(e)}")

    @classmethod
    def get_model(cls, model_name):
        return cls._loaded_models.get(model_name)