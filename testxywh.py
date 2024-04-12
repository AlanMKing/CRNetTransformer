from detrdataset import *
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

image_dir = r'F:\BiYeSheJi\program\Dataset\Images\Train'
gt_dirs = [
    r'F:\BiYeSheJi\program\Transformer\totaltext\polys',
    r'F:\BiYeSheJi\program\Transformer\totaltext\Tf',
    r'F:\BiYeSheJi\program\Transformer\totaltext\Tc',
    r'F:\BiYeSheJi\program\Transformer\totaltext\Xoffset',
    r'F:\BiYeSheJi\program\Transformer\totaltext\Yoffset'
]


def convert_cxcywh_to_xyxy(boxes):
    """
    将边界框从 (cx, cy, w, h) 格式转换为 (x_min, y_min, x_max, y_max) 格式。
    """
    cx, cy, w, h = boxes.unbind(-1)
    x_min = cx - 0.5 * w
    y_min = cy - 0.5 * h
    x_max = cx + 0.5 * w
    y_max = cy + 0.5 * h
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)

def visualize_boxes(image_tensor, boxes):
    """
    可视化图像上的边界框。
    """
    # 将图像张量转换为 NumPy 数组，并调整通道顺序
    image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

    # 转换边界框格式
    boxes = convert_cxcywh_to_xyxy(boxes)

    # 创建一个图和一个坐标轴
    fig, ax = plt.subplots(1)
    # 显示图像
    ax.imshow(image)

    # 为每个边界框添加一个矩形
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # 显示结果
    plt.show()
# 假设你有一个图像和一个边界框张量
# image = plt.imread('your_image_path.jpg')  # 替换为你的图像路径
# boxes = torch.tensor([[[100, 150, 50, 80], [200, 250, 60, 90]]])  # 示例边界框

# 可视化边界框
# visualize_boxes(image, boxes[0])  # 传入第一个批次的边界框

dataset = DETRDataset(image_dir, gt_dirs, target_size=(160,160))
dataloader = DataLoader(dataset,1)
for img,gts in dataloader:
    boxes = gts['boxes'].cpu()
    # 将图像张量转换为 NumPy 数组，并调整通道顺序
    # image = img.squeeze().permute(1, 2, 0).numpy()
    visualize_boxes(img, boxes[0])

