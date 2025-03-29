"""

"""
from django.shortcuts import render
from django.http import JsonResponse
from model_manager.model_manage import ModelManager


def classify_image(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({"error": "未上传图片"}, status=400)

        # 获取分类模型
        model = ModelManager.get_model("classification")
        if not model:
            return JsonResponse({"error": "模型未加载"}, status=500)

        try:
            # 执行推理
            result = model.infer(image_file)
            return JsonResponse()
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "仅支持POST请求"}, status=405)