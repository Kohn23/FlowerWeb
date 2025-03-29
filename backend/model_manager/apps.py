from django.apps import AppConfig


class ModelManagerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "model_manager"

    def ready(self):
        # 仅在主进程中加载模型（避免多Worker重复加载）
        import os
        if os.environ.get('RUN_MAIN') == 'true':
            from .model_manage import ModelManager
            ModelManager.load_all_models()
