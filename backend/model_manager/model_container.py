"""
    1、这里编写模型的容器类，负责加载、数据处理、推理等实现
"""

import onnxruntime
import logging
import numpy as np

from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = None

    def load_model(self):
        """
            加载ONNX模型
        """
        try:
            self.session = onnxruntime.InferenceSession(self.model_path)
            logger.info(f"成功加载模型: {self.model_path}")
        except Exception as e:
            logger.error(f"模型加载失败: {self.model_path}, 错误: {str(e)}")
            raise

    @abstractmethod
    def preprocess(self, input_data):
        """
            预处理输入数据（由子类实现）
        """
        pass

    @abstractmethod
    def postprocess(self, output_data):
        """
            后处理输出数据（由子类实现）
        """
        pass

    def infer(self, input_data):
        """
            执行推理
        """
        if not self.session:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        # 预处理
        processed_input = self.preprocess(input_data)

        # ONNX推理
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        output = self.session.run([output_name], {input_name: processed_input})[0]

        # 后处理
        return self.postprocess(output)


class ClassificationModel(BaseModel):
    def preprocess(self, image):
        # 示例：将图像转换为模型需要的输入格式
        # 假设输入为归一化的RGB图像数组
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        return image_array.astype(np.float32)

    def postprocess(self, output):
        # 示例：将模型输出转换为类别标签
        class_id = np.argmax(output)
        return {"class_id": int(class_id), "confidence": float(output[class_id])}
