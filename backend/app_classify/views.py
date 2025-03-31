"""
    这里处理前端发送回来的
"""
from django.shortcuts import render
from django.http import JsonResponse
from model_manager.model_manage import ModelManager
from .utils.Response import StandardResponse


def classify_image(request):
    if request.method == 'POST':
        image_file = request.FILES.get('flower_img')
        if not image_file:
            return StandardResponse(status=400, success=False, data={"error": "未上传图片"})

        # 获取分类模型
        model = ModelManager.get_model("classification")
        if not model:
            return StandardResponse(status=500, success=False, data={"error": "模型未加载"})

        try:
            # 执行推理
            result = model.infer(image_file)
            return StandardResponse(status=200, data={result})
        except Exception as e:
            return StandardResponse(status=500, success=False, data={"error": str(e)})
    return StandardResponse(status=405, success=False, data={"error": "仅支持POST请求"})