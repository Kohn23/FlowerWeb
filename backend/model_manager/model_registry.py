"""
    1、在这里管理注册的（模型类型）
    2、在settings里面使用MODEL_PATH管理具体使用（模型的路径）
"""

from django.conf import settings
from model_container import ClassificationModel


class ModelRegistry:
    _models = {
        "classification": {
            "class": ClassificationModel,
            "model_path": settings.MODEL_PATHS["classification"]
        }
        # 未来的模型在这里添加
    }

    @classmethod
    def get_model_config(cls, model_name):
        return cls._models.get(model_name)