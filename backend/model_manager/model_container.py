"""
    1、这里编写模型的容器类，负责加载、数据处理、推理等实现
"""

import onnxruntime
import logging
import numpy as np
import torch
import torchvision.transforms as transforms

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

# 分类标签
class_mapping = {
    '21': 'fire lily', '3': 'canterbury bells', '45': 'bolero deep blue', '1': 'pink primrose',
    '34': 'mexican aster', '27': 'prince of wales feathers', '7': 'moon orchid', '16': 'globe-flower',
    '25': 'grape hyacinth', '26': 'corn poppy', '79': 'toad lily', '39': 'siam tulip', '24': 'red ginger',
    '67': 'spring crocus', '35': 'alpine sea holly', '32': 'garden phlox', '10': 'globe thistle',
    '6': 'tiger lily', '93': 'ball moss', '33': 'love in the mist', '9': 'monkshood', '102': 'blackberry lily',
    '14': 'spear thistle', '19': 'balloon flower', '100': 'blanket flower', '13': 'king protea', '49': 'oxeye daisy',
    '15': 'yellow iris', '61': 'cautleya spicata', '31': 'carnation', '64': 'silverbush', '68': 'bearded iris',
    '63': 'black-eyed susan', '69': 'windflower', '62': 'japanese anemone', '20': 'giant white arum lily',
    '38': 'great masterwort', '4': 'sweet pea', '86': 'tree mallow', '101': 'trumpet creeper', '42': 'daffodil',
    '22': 'pincushion flower', '2': 'hard-leaved pocket orchid', '54': 'sunflower', '66': 'osteospermum',
    '70': 'tree poppy', '85': 'desert-rose', '99': 'bromelia', '87': 'magnolia', '5': 'english marigold',
    '92': 'bee balm', '28': 'stemless gentian', '97': 'mallow', '57': 'gaura', '40': 'lenten rose',
    '47': 'marigold', '59': 'orange dahlia', '48': 'buttercup', '55': 'pelargonium', '36': 'ruby-lipped cattleya',
    '91': 'hippeastrum', '29': 'artichoke', '71': 'gazania', '90': 'canna lily', '18': 'peruvian lily',
    '98': 'mexican petunia', '8': 'bird of paradise', '30': 'sweet william', '17': 'purple coneflower',
    '52': 'wild pansy', '84': 'columbine', '12': "colt's foot", '11': 'snapdragon', '96': 'camellia',
    '23': 'fritillary', '50': 'common dandelion', '44': 'poinsettia', '53': 'primula', '72': 'azalea',
    '65': 'californian poppy', '80': 'anthurium', '76': 'morning glory', '37': 'cape flower',
    '56': 'bishop of llandaff', '60': 'pink-yellow dahlia', '82': 'clematis', '58': 'geranium',
    '75': 'thorn apple', '41': 'barbeton daisy', '95': 'bougainvillea', '43': 'sword lily', '83': 'hibiscus',
    '78': 'lotus lotus', '88': 'cyclamen', '94': 'foxglove', '81': 'frangipani', '74': 'rose',
    '89': 'watercress', '73': 'water lily', '46': 'wallflower', '77': 'passion flower', '51': 'petunia'
}

sorted_class_names = sorted(class_mapping.values())

# 预处理(晓雨提供)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 固定尺寸 (224,224)
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


class ClassificationModel(BaseModel):
    def preprocess(self, image):
        # 转换并确保输出为NumPy数组
        processed_tensor = transform(image).unsqueeze(0).numpy()
        return processed_tensor

    def postprocess(self, output):
        # 关键修改3：直接处理NumPy数组
        output_tensor = torch.from_numpy(output)
        top5_probs = torch.nn.functional.softmax(output_tensor, dim=1)
        top5_values, top5_indices = torch.topk(top5_probs, 5)
        top5_results = [(sorted_class_names[idx], prob.item()) for idx, prob in zip(top5_indices[0], top5_values[0])]
        return top5_results