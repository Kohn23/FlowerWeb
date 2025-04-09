from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
import uuid
from model_manager.model_style_transfer import StyleTransferModel
from .utils.Response import StandardResponse


def generate_image(request):
    if request.method == 'POST':
        # 获取上传的图片
        content_image = request.FILES.get('contentImg')
        style_image = request.FILES.get('styleImg')

        if not content_image or not style_image:
            return StandardResponse(status=400, success=False)

        try:
            # 保存上传的图片到 media 文件夹
            content_path = os.path.join('uploads', f'content_{uuid.uuid4()}.jpg')
            style_path = os.path.join('uploads', f'style_{uuid.uuid4()}.jpg')

            default_storage.save(content_path, ContentFile(content_image.read()))
            default_storage.save(style_path, ContentFile(style_image.read()))

            # 初始化风格迁移模型
            model = StyleTransferModel()

            # 生成风格迁移后的图片
            output_path = os.path.join('generated', f'style_transfer_{uuid.uuid4()}.jpg')
            output_full_path = os.path.join(settings.MEDIA_ROOT, output_path)

            # 调用风格迁移
            processing_time = model.transfer_style(
                content_path=os.path.join(settings.MEDIA_ROOT, content_path),
                style_path=os.path.join(settings.MEDIA_ROOT, style_path),
                output_path=output_full_path,
                num_steps=100,
                style_weight=100000,
                content_weight=1,
                image_size=512
            )

            # 构建生成图片的 URL
            image_url = os.path.join(settings.MEDIA_URL, output_path)

            # 返回结果
            return StandardResponse(url=image_url)

        except Exception as e:
            return StandardResponse(status=500, success=False)

    return render(request, 'upload.html')
