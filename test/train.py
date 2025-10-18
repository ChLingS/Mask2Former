import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from dataset import CroppedRemoteSensingDataset  # 导入自定义数据集类
from net import ImprovedUNet


# 计算IoU（交并比）
def calculate_iou(pred, target):
    pred = torch.argmax(pred, dim=1)  # 获取预测类别
    intersection = torch.logical_and(pred, target).sum()
    union = torch.logical_or(pred, target).sum()
    iou = (intersection + 1e-6) / (union + 1e-6)  # 添加小值避免除零
    return iou.item()

# 保存模型检查点
def save_checkpoint(model, optimizer, epoch, loss, iou, is_best=False):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
        'iou': iou
    }
    filename = f"unet_checkpoint_epoch{epoch}.pth"
    torch.save(state, filename)
    
    if is_best:
        torch.save(state, "unet_best_model.pth")

# 加载模型检查点
def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        print(f"=> 加载检查点 '{filename}'")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> 加载完成 (epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f}, IoU {checkpoint['iou']:.4f})")
        return checkpoint['epoch']
    else:
        print(f"=> 未找到检查点 '{filename}'")
        return 0

if __name__ == "__main__":
    # 配置参数
    img_dir = r"F:\yanRuyou"
    mask_dir = r"E:\帮人做东西\严茹尤\训练数据\grids"
    output_dir = r"F:\mask2formerMiddata"
    batch_size = 2
    num_epochs = 50
    learning_rate = 1e-3
    val_split = 0.2  # 20%的数据用于验证
    
    # 创建输出目录
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet归一化
    ])
    
    # 创建完整数据集
    full_dataset = CroppedRemoteSensingDataset(
        img_dir = img_dir, 
        mask_dir = mask_dir, 
        crop_size=512, 
        augment=True,
        save_cropped=True,
        output_dir=output_dir,
        min_valid_ratio = 0.1,
        pre_cropped=True,
    )
    
    # 分割训练集和验证集
    val_size = int(val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")
    
    # 初始化模型
    model = ImprovedUNet(in_channels=3, out_channels=2).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(device))  # 给地块类别更高权重
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose='True')
    
    # 加载现有检查点（如果存在）
    start_epoch = load_checkpoint(model, optimizer, "unet_best_model.pth")
    
    # 训练循环
    best_iou = 0.0
    train_losses, val_losses, val_ious = [], [], []
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"训练 {epoch+1}/{num_epochs}")
        
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)
        print(f"训练损失: {epoch_loss:.4f}")
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                
                # 计算IoU
                val_iou += calculate_iou(outputs, masks) * images.size(0)
        
        val_loss /= len(val_dataset)
        val_iou /= len(val_dataset)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        
        print(f"验证损失: {val_loss:.4f}, IoU: {val_iou:.4f}")
        
        # 更新学习率
        scheduler.step(val_iou)
        
        # 保存检查点
        save_checkpoint(model, optimizer, epoch+1, val_loss, val_iou)
        
        # 保存最佳模型
        if val_iou > best_iou:
            best_iou = val_iou
            save_checkpoint(model, optimizer, epoch+1, val_loss, val_iou, is_best=True)
            print(f"新的最佳模型保存，IoU: {best_iou:.4f}")
    
    # 保存最终模型
    torch.save(model.state_dict(), "unet_field_final.pth")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_ious, label='验证IoU')
    plt.title('验证IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()
    
    print("训练完成，模型已保存。")