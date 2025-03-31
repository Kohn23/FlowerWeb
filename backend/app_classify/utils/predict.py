import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn

# --------------- 1. 载入模型 ---------------
class ResNetExtractor(nn.Module):
    def __init__(self, out_dim=256):
        super(ResNetExtractor, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, out_dim)

    def forward(self, x):
        return self.model(x)

class SmallVisionTransformer(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_heads=2,
                 num_layers=2, num_classes=102, dropout=0.3):
        super(SmallVisionTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 2, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [B, 256]
        x = self.embedding(x).unsqueeze(1)  # [B, 1, hidden_dim]
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, hidden_dim]
        x = torch.cat([cls_tokens, x], dim=1)          # [B, 2, hidden_dim]
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x[:, 0, :]  # 取 CLS token
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# --------------- 2. 加载训练好的权重 ---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 类别索引到名称的映射（按照字典序排序）
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

# 加载模型并加载权重（使用与训练时一致的配置）
resnet = ResNetExtractor(out_dim=256).to(device)
vit_model = SmallVisionTransformer(input_dim=256, hidden_dim=128, num_heads=2,
                                   num_layers=2, num_classes=len(sorted_class_names), dropout=0.3).to(device)

checkpoint = torch.load('./weight/epoch_24_res18.pth', map_location=device)
resnet.load_state_dict(checkpoint['resnet'])
vit_model.load_state_dict(checkpoint['vit_model'])

resnet.eval()
vit_model.eval()

# --------------- 3. 预处理输入图像 ---------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 固定尺寸 (224,224)
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(image)         # 得到 256 维特征
        output = vit_model(features)       # ViT 得到预测输出
        probs = torch.nn.functional.softmax(output, dim=1)
        top5_probs, top5_indices = torch.topk(probs, 5)
        top5_results = [(sorted_class_names[idx], prob.item()) for idx, prob in zip(top5_indices[0], top5_probs[0])]
    return top5_results

# --------------- 4. 进行预测 ---------------
image_path = "c8(1)(1).jpg"
top5_predictions = predict(image_path)

print("Top 5 Predictions:")
for i, (class_name, prob) in enumerate(top5_predictions, 1):
    print(f"{i}. {class_name} - {prob:.2%}")
