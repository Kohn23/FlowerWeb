"""
    这里处理前端发送回来的
"""
from django.shortcuts import render
from django.http import JsonResponse
from model_manager.model_manage import ModelManager
from .utils.Response import StandardResponse
from PIL import Image  # 需要安装Pillow库
from io import BytesIO


def classify_image(request):
    if request.method == 'POST':
        image_file = request.FILES.get('flower_img')
        if not image_file:
            return StandardResponse(status=400, success=False, data={"error": "未上传图片"})

        try:
            # 将上传文件转换为PIL图像对象
            img = Image.open(BytesIO(image_file.read())).convert("RGB")

            # 获取分类模型
            model = ModelManager.get_model("classification")
            if not model:
                return StandardResponse(status=500, success=False, data={"error": "模型未加载"})

            # 推理
            result = model.infer(img)
            return StandardResponse(status=200, data=result)

        except IOError:
            return StandardResponse(status=400, success=False, data={"error": "无效的图片文件"})
        except Exception as e:
            return StandardResponse(status=500, success=False, data={"error": str(e)})

    return StandardResponse(status=405, success=False, data={"error": "仅支持POST请求"})