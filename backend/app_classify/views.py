"""
    这里处理前端发送回来的
"""
import requests
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
            flower = max(result, key=lambda x: x[1])[0]

            flower_details = generate_flower_details(flower)
            return StandardResponse(status=200, data={"matches": result, "details": flower_details})

        except IOError:
            return StandardResponse(status=400, success=False, data={"error": "无效的图片文件"})
        except Exception as e:
            return StandardResponse(status=500, success=False, data={"error": str(e)})

    return StandardResponse(status=405, success=False, data={"error": "仅支持POST请求"})


# def generate_flower_details(flower_type):
#     """
#     调用 DeepSeek API 生成花语
#     """
#     # DeepSeek API 配置
#     api_url = "https://api.deepseek.com/v1/chat/completions"
#     api_key = "sk-fbf17fe2eaec42879d3224a0aa0b66a8"  # 替换为你的 DeepSeek API Key
#
#     # 请求头
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {api_key}"
#     }
#
#     # 请求体
#     data = {
#         "model": "deepseek-chat",
#         "messages": [
#             {
#                 "role": "system",
#                 "content": "你是一个花语生成器，根据花的种类生成一段花语。"
#             },
#             {
#                 "role": "user",
#                 "content": f"根据 {flower_type} 生成一段花语。"
#             }
#         ]
#     }
#
#     try:
#         # 发送请求
#         response = requests.post(api_url, headers=headers, json=data)
#         response.raise_for_status()  # 检查请求是否成功
#
#         # 解析响应
#         response_data = response.json()
#         flower_language = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
#
#         print(flower_language)
#         return flower_language
#
#     except requests.exceptions.RequestException as e:
#         return f"API 调用失败: {str(e)}"


def generate_flower_details(flower_type):
    """
    调用 DeepSeek API 生成结构化花卉信息
    """
    api_url = "https://api.deepseek.com/v1/chat/completions"
    api_key = "sk-fbf17fe2eaec42879d3224a0aa0b66a8"  # 替换为你的 DeepSeek API Key

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # 修改后的提示词要求结构化输出
    structured_prompt = f"""请为{flower_type}生成结构化信息，包含以下字段（保持英文key）：
    - sunlight（光照需求）
    - water（浇水频率）
    - temperature（适宜温度，带°C单位）
    - season（主要开花季节）
    - soil（土壤类型）
    - height（典型高度，带cm单位）
    - spread（扩展范围，带cm单位）
    - lifespan（生命周期）
    用JSON格式返回，不要包含额外说明。示例格式：
    {{
        "sunlight": "Full Sun",
        "water": "Regular",
        "temperature": "15-25°C",
        "season": "Spring",
        "soil": "Well-drained",
        "height": "30-60cm",
        "spread": "20-40cm",
        "lifespan": "Perennial"
    }}"""

    data = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "你是一个植物学专家，需要严格按照用户要求的格式返回结构化数据"
            },
            {
                "role": "user",
                "content": structured_prompt
            }
        ]
    }

    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()

        # 提取并解析JSON内容
        response_data = response.json()
        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # 尝试解析返回的JSON字符串
        import json
        structured_data = json.loads(content)

        # 确保包含所有必需字段
        required_fields = ["sunlight", "water", "temperature", "season",
                           "soil", "height", "spread", "lifespan"]
        return {field: structured_data.get(field, "N/A") for field in required_fields}

    except json.JSONDecodeError:
        return {"error": "API返回格式解析失败"}
    except requests.exceptions.RequestException as e:
        return {"error": f"API调用失败: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}