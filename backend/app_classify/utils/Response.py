"""
    这里通过继承JsonResponse类来实现对返回前端的Json格式的控制
"""

from django.http import JsonResponse


class StandardResponse(JsonResponse):
    def __init__(self, status=200, code=0, success=True, data=None, **kwargs):
        response_data = {
            "code": code,
            "success": success,
            "data": data,
        }
        super().__init__(data=response_data, status=status, **kwargs)
