"""
测试模型可视化部分，经过几次模型修改后已经不匹配
详细见jupyter notebook版本
"""

import torch
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from config import *
from crnetdetr import *

# 加载模型
model_path = r'F:\BiYeSheJi\program\crnetdetr\detr_model30.pth'
model = CRNETDETR().to(device)
model = torch.load(model_path).to(device)
# model = Transformer(N=200, batch_size=1, d_model=256, num_box=200).to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((640, 640),antialias=True),
    transforms.ToTensor()
])

# 准备输入图片
image_path = r"F:\BiYeSheJi\program\Dataset\Images\Train\img188.jpg"
image = Image.open(image_path).convert('RGB')
# 计算缩放比例
scale_width, scale_height = image.width / 640., image.height / 640.
input_tensor = transform(image).unsqueeze(0).to(device)


# 模型推理
with torch.no_grad():
    out = model(input_tensor)

print(out)
# 阈值筛选
confidence_threshold = 0.1
text_indices = (out['pred_logits'][:, :, 0] > confidence_threshold).nonzero(as_tuple=True)[1]

# 提取文本框坐标
text_boxes = out['pred_boxes'][0, text_indices, :].cpu().numpy()

# 可视化
plt.figure(figsize=(10, 10))
plt.imshow(image)
for box in text_boxes:
    cx, cy, w, h = box[:4]  # 假设坐标顺序为 [x_min, y_min, x_max, y_max]
    # orig_cx,orig_cy = cx * scale_width,cy * scale_width
    # orig_w,orig_h = w * scale_width, h * scale_width
    orig_cx,orig_cy = cx,cy
    orig_w,orig_h = w, h
    xmin = orig_cx - orig_w / 2
    ymin = orig_cy - orig_h / 2
    xmax = orig_cx + orig_w / 2
    ymax = orig_cy + orig_h / 2
    plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='red', linewidth=2))
plt.axis('off')
plt.show()
