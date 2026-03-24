# 这个类会自动识别人物 ID（1-16 为训练，17-25 为测试）
# 并按照 10 -> 20（输入 10 帧，预测 20 帧）的滑动窗口提取序列。

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms

class KTHDataset(Dataset):
    def __init__(self, data_root, mode='train', T_in=10, T_out=20, image_size=128):
        self.data_root = data_root
        self.mode = mode
        self.T_total = T_in + T_out
        self.image_size = image_size
        self.samples = []

        # 严格遵守 SimVP 论文的人物划分协议
        train_persons = [f'person{i:02d}' for i in range(1, 17)]
        test_persons = [f'person{i:02d}' for i in range(17, 26)]
        target_persons = train_persons if mode == 'train' else test_persons

        # 扫描 6 个动作目录
        actions = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
        
        for action in actions:
            action_dir = os.path.join(data_root, action)
            if not os.path.exists(action_dir): continue
            
            # 获取该动作下的所有视频文件夹 (avi 目录)
            video_folders = sorted(os.listdir(action_dir))
            for folder in video_folders:
                # 检查人物是否属于当前 mode (train/test)
                person_id = folder.split('_')[0]
                if person_id in target_persons:
                    folder_path = os.path.join(action_dir, folder)
                    imgs = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
                    
                    # 使用滑动窗口提取序列，步长为 1 (与 SimVP 保持一致以获得最大数据量)
                    if len(imgs) >= self.T_total:
                        for i in range(len(imgs) - self.T_total + 1):
                            self.samples.append([os.path.join(folder_path, imgs[j]) for j in range(i, i + self.T_total)])

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(), # KTH 默认为灰度训练
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_paths = self.samples[idx]
        frames = []
        for p in img_paths:
            img = Image.open(p)
            frames.append(self.transform(img))
        
        # 转换后维度: [T, C, H, W]
        data = torch.stack(frames) 
        # 返回 (输入, 标签) -> (10帧, 20帧)
        return data[:10], data[10:]


def load_kth(batch_size, val_batch_size, data_root, num_workers):
    print(f"⏳ 正在加载 KTH 数据集 (输入 10 帧 -> 预测 20 帧): {data_root}")
    
    train_dataset = KTHDataset(data_root, mode='train', T_in=10, T_out=20, image_size=128)
    test_dataset = KTHDataset(data_root, mode='test', T_in=10, T_out=20, image_size=128)
    
    print(f"✅ KTH 加载成功! 训练集: {len(train_dataset)} 序列, 测试集: {len(test_dataset)} 序列")

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_test = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    # KTH 的 transforms.ToTensor() 已经将其缩放到了 [0, 1] 范围
    mean, std = 0, 1  
    
    return dataloader_train, dataloader_test, dataloader_test, mean, std