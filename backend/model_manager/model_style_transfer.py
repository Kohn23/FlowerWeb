"""

# 2. model/model_style_transfer.py - 风格迁移模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import time
import os


class StyleTransferModel:
    def __init__(self, device=None):
        # 如果没有指定设备，检查CUDA可用性
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # 加载VGG19模型并转移到指定设备
        self.cnn = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(self.device).eval()
        
        # 标准化参数
        self.normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
    
    def image_loader(self, image_path, image_size=None):
        """加载图像并转换为张量"""
        image = Image.open(image_path)
        
        # 如果指定了尺寸，调整图像大小
        if image_size is not None:
            loader = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
        else:
            loader = transforms.Compose([
                transforms.ToTensor()
            ])
        
        # 添加批次维度
        image = loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)
    
    def save_image(self, tensor, path):
        """将张量保存为图像文件"""
        image = tensor.cpu().clone()
        image = image.squeeze(0)  # 移除批次维度
        unloader = transforms.ToPILImage()
        image = unloader(image)
        image.save(path)
    
    def gram_matrix(self, input_tensor):
        """计算Gram矩阵"""
        batch_size, channels, height, width = input_tensor.shape
        features = input_tensor.view(batch_size * channels, height * width)
        G = torch.mm(features, features.t())
        # 归一化Gram矩阵
        return G.div(batch_size * channels * height * width)
    
    def get_style_model_and_losses(self, style_img, content_img):
        """构建风格迁移模型和损失函数"""
        # 内容层和风格层
        content_layers = ['conv_4']
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
        # 创建标准化模块
        normalization = Normalization(self.normalization_mean, self.normalization_std).to(self.device)
        
        # 存储内容损失和风格损失
        content_losses = []
        style_losses = []
        
        # 顺序模型
        model = nn.Sequential(normalization)
        
        i = 0  # 卷积层计数器
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                # 替换为非原地版本
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')
            
            model.add_module(name, layer)
            
            # 添加内容损失
            if name in content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module(f"content_loss_{i}", content_loss)
                content_losses.append(content_loss)
            
            # 添加风格损失
            if name in style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature, self.gram_matrix)
                model.add_module(f"style_loss_{i}", style_loss)
                style_losses.append(style_loss)
        
        # 修剪掉最后一个内容和风格损失之后的层
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        
        model = model[:(i + 1)]
        
        return model, style_losses, content_losses
    
    def transfer_style(self, content_path, style_path, output_path, num_steps=100, 
                       style_weight=1000000, content_weight=1, image_size=512):
        """执行风格迁移"""
        # 记录开始时间
        start_time = time.time()
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 加载内容和风格图像
        content_img = self.image_loader(content_path, image_size)
        style_img = self.image_loader(style_path, image_size)
        
        # 确保风格图像与内容图像大小一致
        if style_img.shape[2:] != content_img.shape[2:]:
            style_img = F.interpolate(style_img, size=content_img.shape[2:], 
                                    mode='bilinear', align_corners=False)
        
        # 创建优化的输入图像（初始值为内容图像）
        input_img = content_img.clone()
        
        # 获取模型和损失
        model, style_losses, content_losses = self.get_style_model_and_losses(style_img, content_img)
        
        # 设置优化器
        input_img.requires_grad_(True)
        optimizer = optim.LBFGS([input_img])
        
        # 优化过程
        print(f"开始优化过程: {num_steps}步...")
        run = [0]
        
        while run[0] <= num_steps:
            def closure():
                # 确保值在[0, 1]范围内
                with torch.no_grad():
                    input_img.clamp_(0, 1)
                
                optimizer.zero_grad()
                model(input_img)
                
                style_score = 0
                content_score = 0
                
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                
                # 加权损失
                style_score *= style_weight
                content_score *= content_weight
                
                loss = style_score + content_score
                loss.backward()
                
                run[0] += 1
                if run[0] % 20 == 0:
                    print(f"步骤 {run[0]}/{num_steps}")
                    print(f'风格损失: {style_score.item():.4f}, 内容损失: {content_score.item():.4f}')
                
                return loss
            
            optimizer.step(closure)
        
        # 确保最终图像在[0,1]范围内
        with torch.no_grad():
            input_img.clamp_(0, 1)
        
        # 保存结果图像
        self.save_image(input_img, output_path)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        print(f"风格迁移完成! 耗时: {processing_time:.2f} 秒")
        
        return processing_time


# 标准化模块
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().view(-1, 1, 1)
        self.std = std.clone().view(-1, 1, 1)
        
    def forward(self, img):
        return (img - self.mean) / self.std


# 内容损失类
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = None
        
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


# 风格损失类
class StyleLoss(nn.Module):
    def __init__(self, target_feature, gram_fn):
        super(StyleLoss, self).__init__()
        self.target = gram_fn(target_feature).detach()
        self.gram_fn = gram_fn
        self.loss = None
        
    def forward(self, input):
        G = self.gram_fn(input)
        self.loss = F.mse_loss(G, self.target)
        return input