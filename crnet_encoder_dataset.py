"""
Dataset，gts返回类型为字典，Tf Tc Xoffset Yoffset 由getlabel生成并存储为npy
"""

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import cv2
from config import *


class CRNETENCODERDataset(Dataset):
    def __init__(self, image_dir, gt_dirs, target_size=(128, 128)):
        self.image_dir = image_dir
        self.gt_dirs = gt_dirs
        self.transform = transforms.Compose([transforms.Resize((512, 512), antialias=True), transforms.ToTensor()])
        self.target_size = target_size
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        # width, height = image.size
        image = self.transform(image)

        gts = {'Tf':None,'Tc':None,'Xoffset':None,'Yoffset':None}
        for gt_dir in self.gt_dirs:
            gt_path = os.path.join(gt_dir, self.image_files[idx].replace('.jpg', '.npy'))
            gt_data = np.load(gt_path, allow_pickle=True)
            gt_resized = cv2.resize(gt_data, self.target_size, interpolation=cv2.INTER_NEAREST)
            gt_tensor = torch.tensor(gt_resized, dtype=torch.float32)
            gt_key = gt_dir.split("/")[-1]  # Assuming the key is the last part of the directory name
            gts[gt_key] = gt_tensor

        return image,gts

# image_dir = r'..\Images\Train'
# gt_dirs = [
#     r'..\Tf',
#     r'..\Tc',
#     r'..\Xoffset',
#     r'..\Yoffset'
# ]
#
# dataset = DETRDataset(image_dir, gt_dirs, target_size=(128,128))
# dataloader = DataLoader(dataset,1)
# for img,gts in dataloader:
#     print('x')
# print('x')
