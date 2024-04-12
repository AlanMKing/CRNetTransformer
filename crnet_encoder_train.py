from config import *
from crnet_encoder_model import CRNETENCODER
from crnet_encoder_dataset import CRNETENCODERDataset
from crnet_encoder_loss import CRNETENCODERLoss



image_dir = r'F:\BiYeSheJi\program\Dataset\Images\Train'
gt_dirs = [
    r'F:\BiYeSheJi\program\Transformer\totaltext\Tf',
    r'F:\BiYeSheJi\program\Transformer\totaltext\Tc',
    r'F:\BiYeSheJi\program\Transformer\totaltext\Xoffset',
    r'F:\BiYeSheJi\program\Transformer\totaltext\Yoffset'
]

def train_model():
    # 超参数设置
    num_epochs = 100
    learning_rate = 0.01
    batch_size = 2

    # 加载数据集
    dataset = CRNETENCODERDataset(image_dir, gt_dirs, target_size=(128, 128))  # target_size=(160, 160) 用于加载crnet的gt计算loss，目前未使用
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # ,collate_fn=custom_collate_fn
    # 这里不太会使用dataloader，所以没有做custom_collate_fn，计算loss时格式不匹配可能来自这里，但detr官方使用coco格式，不能作为参考
    # 定义模型
    model = CRNETENCODER().to(device)
    # 匈牙利匹配实例化
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
            # loss = loss_fn(pre_class, pre_box, tf, tc, xoffset, yoffset, gt)
            loss_dict = loss_fn(outputs, gts)
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            total_loss += losses.item()

            # 反向传播和优化
            losses.backward()

            # 梯度裁剪
            # torch.nn.utils.clip_grad_norm_(detr.parameters(), max_norm=1.0)

            optimizer.step()

            end_time = time.time()
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(dataloader)}], Loss: {losses.item():.4f} , Total Time: {end_time - start_time}s")
                # torch.save(transformer, f'trans_model{epoch + 1}.pth')

        end_epoch_time = time.time()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {total_loss / len(dataloader):.4f}, 1 Epoch Time {end_epoch_time - start_time} s")
        if (epoch + 1) % 1 == 0:
            torch.save(model, f'detr_model{epoch+1}.pth')
            # print(outputs['pred_logits'])
            # print(outputs['pred_boxes'])
            pass
        print('***********************************************************************')

    print("Training complete.")


train_model()
