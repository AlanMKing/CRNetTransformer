# 调试使用，具体见jupyter notebook

from config import *
from crnet_encoder_model import CRNETENCODER
from crnet_encoder_dataset import CRNETENCODERDataset
from crnet_encoder_loss import CRNETENCODERLoss



image_dir = r'...\Dataset\Images\Train'
gt_dirs = [
    r'..\Tf',
    r'..\Tc',
    r'..\Xoffset',
    r'..\Yoffset'
]

def train_model():
    # 超参数设置
    num_epochs = 100
    learning_rate = 0.01
    batch_size = 2

    # 加载数据集
    dataset = CRNETENCODERDataset(image_dir, gt_dirs, target_size=(128, 128))  # target_size=(160, 160) 用于加载crnet的gt计算loss，目前未使用
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 定义模型
    model = CRNETENCODER().to(device)

    # 定义损失函数参数
    weight_dict = {'loss_all': 2, 'loss_crnet': 0.5}
    loss_fn = CRNETENCODERLoss()
    loss_fn.to(device)

    # 训练模型
    for epoch in range(num_epochs):
        if epoch > 10:
            learning_rate *= 0.5
        elif epoch > 20:
            learning_rate *= 0.5
        # 定义优化器
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        model.train()

        total_loss = 0
        start_time = time.time()
        print('***********************************************************************')
        print(f'start training epoch{epoch + 1}')
        for batch_idx, (images, gts) in enumerate(dataloader):
            images = images.to(device)
            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss_dict = loss_fn(outputs, gts)
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            total_loss += losses.item()

            # 反向传播和优化
            losses.backward()

            # 梯度裁剪
            # torch.nn.utils.clip_grad_norm_(detr.parameters(), max_norm=1.0)

            # 梯度消失检查（略）

            optimizer.step()

            end_time = time.time()
            if (batch_idx + 1) % 50 == 0:
                print(loss_dict)
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(dataloader)}], Loss: {losses.item():.4f} , Total Time: {end_time - start_time}s")

        end_epoch_time = time.time()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {total_loss / len(dataloader):.4f}, 1 Epoch Time {end_epoch_time - start_time} s")
        if (epoch + 1) % 5 == 0:
            torch.save(model, f'detr_model{epoch+1}.pth')
            pass
        print('***********************************************************************')

    print("Training complete.")


train_model()
