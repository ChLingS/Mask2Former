import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
from typing import Optional, Callable, List, Tuple, Any, Dict
from pathlib import Path

class CroppedRemoteSensingDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        crop_size: int = 256,
        transform: Optional[Callable] = None,
        augment: bool = False,
        save_cropped: bool = False,
        output_dir: Optional[str] = None,
        min_valid_pixels: int = 100,
        min_valid_ratio: float = 0.01,
        pre_cropped: bool = False
    ):
        """
        裁剪遥感数据集类
        
        参数:
            img_dir: 图像目录
            mask_dir: 掩码目录
            crop_size: 裁剪尺寸
            transform: 数据变换
            augment: 是否数据增强
            save_cropped: 是否保存裁剪后的中间文件
            output_dir: 中间文件输出目录
            min_valid_pixels: 掩码中有效像素的最小数量
            min_valid_ratio: 掩码中有效像素的最小比例
            pre_cropped: 是否直接加载预裁剪的图像块
        """
        # 使用 pathlib 处理路径
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.output_dir = Path(output_dir) if output_dir else None
        self.crop_size = crop_size
        self.transform = transform
        self.augment = augment
        self.save_cropped = save_cropped
        self.min_valid_pixels = min_valid_pixels
        self.min_valid_ratio = min_valid_ratio
        self.pre_cropped = pre_cropped
        
        print(f"图像目录: {self.img_dir}")
        print(f"掩码目录: {self.mask_dir}")
        if self.output_dir:
            print(f"中间文件输出目录: {self.output_dir}")
        print(f"裁剪尺寸: {self.crop_size}")
        print(f"最小有效像素数量: {self.min_valid_pixels}")
        print(f"最小有效像素比例: {self.min_valid_ratio:.2%}")
        print(f"预裁剪模式: {self.pre_cropped}")

        # 创建输出目录
        if save_cropped and self.output_dir:
            (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "masks").mkdir(parents=True, exist_ok=True)
        
        if pre_cropped:
            # 直接加载预裁剪的图像块路径
            self.image_paths, self.mask_paths = self.load_precropped_paths()
        else:
            # 递归查找所有子目录中的tif文件
            self.img_files = list(self.img_dir.rglob("*.tif"))
            self.mask_files = list(self.mask_dir.rglob("*.tif"))
            
            # 匹配图像和掩码文件
            self.matching_pairs = self.match_dataset(self.img_files, self.mask_files)
            
            print(f"找到 {len(self.matching_pairs)} 对匹配的图像-掩码文件")
            
            # 预裁剪并保存文件路径
            self.image_paths, self.mask_paths = self.preload_and_crop()
        
        print(f"加载 {len(self.image_paths)} 个图像块路径")
        
        # 数据增强
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
    
    def load_precropped_paths(self) -> Tuple[List[Path], List[Path]]:
        """直接加载预裁剪的图像块路径"""
        if not self.output_dir:
            raise ValueError("当 pre_cropped=True 时，必须提供 output_dir")
        
        # 获取图像和掩码目录路径
        img_dir_path = self.output_dir / "images"
        mask_dir_path = self.output_dir / "masks"
        
        # 检查目录是否存在
        if not img_dir_path.exists():
            raise FileNotFoundError(f"预裁剪图像目录不存在: {img_dir_path}")
        if not mask_dir_path.exists():
            raise FileNotFoundError(f"预裁剪掩码目录不存在: {mask_dir_path}")
        
        # 获取所有图像和掩码文件
        img_files = sorted(img_dir_path.glob("*.png"))
        mask_files = sorted(mask_dir_path.glob("*.png"))
        
        print(f"找到 {len(img_files)} 个预裁剪图像文件")
        print(f"找到 {len(mask_files)} 个预裁剪掩码文件")
        
        # 创建文件名到路径的映射
        img_dict = {file.name: file for file in img_files}
        mask_dict = {file.name: file for file in mask_files}
        
        # 匹配图像和掩码文件
        image_paths = []
        mask_paths = []
        matched_count = 0
        
        for filename in img_dict:
            if filename in mask_dict:
                image_paths.append(img_dict[filename])
                mask_paths.append(mask_dict[filename])
                matched_count += 1
            else:
                print(f"警告: 找不到与图像 {filename} 对应的掩码文件")
        
        print(f"成功匹配 {matched_count} 对图像-掩码文件路径")
        return image_paths, mask_paths
    
    def match_dataset(self, img_paths, mask_paths):
            """匹配图像和掩码文件"""
            img_dict = {}
            for path in img_paths:
                filename = os.path.basename(path)
                area_name = filename.split('_')[0]
                img_dict[area_name] = path
            
            mask_dict = {}
            for path in mask_paths:
                filename = os.path.basename(path)
                area_name = filename.replace('_mask', '').split('.')[0]
                mask_dict[area_name] = path
            
            matching_pairs = []
            for area_name in img_dict:
                if area_name in mask_dict:
                    matching_pairs.append((img_dict[area_name], mask_dict[area_name]))
                    print(f"匹配成功: {img_dict[area_name]} <-> {mask_dict[area_name]}")
            return matching_pairs
    
    def is_valid_mask(self, mask_tile: np.ndarray) -> bool:
        """检查掩码是否包含足够的值为1的有效像素"""
        # 统计值为1的像素数量
        one_count = np.sum(mask_tile == 1)
        total_pixels = mask_tile.size
        
        if total_pixels == 0:
            return False
        
        # 检查是否满足最小像素数量"或者"最小比例要求
        if one_count >= self.min_valid_pixels or one_count / total_pixels >= self.min_valid_ratio:
            return True
        
        return False
    
    def preload_and_crop(self, max_count: int = 5000) -> Tuple[List[Path], List[Path]]:
        """预裁剪并保存文件路径"""
        image_paths = []
        mask_paths = []
        count_data = 0
        
        for img_path, mask_path in tqdm(self.matching_pairs, desc="处理图像对"):
            try:
                # 读取图像
                with rasterio.open(img_path) as img, rasterio.open(mask_path) as mask:
                    h, w = img.height, img.width
                    
                    # 计算裁剪窗口
                    windows = []
                    for row in range(0, h, self.crop_size):
                        for col in range(0, w, self.crop_size):
                            # 创建窗口对象
                            window = Window(col, row, self.crop_size, self.crop_size)
                            
                            # 检查窗口是否超出图像边界
                            if col + self.crop_size > w or row + self.crop_size > h:
                                continue
                            
                            windows.append(window)
                    
                    # 处理每个窗口
                    for window in tqdm(windows, desc="裁剪窗口", leave=False):
                        # 读取掩码块
                        mask_tile = mask.read(1, window=window)
                        
                        # 检查掩码是否有效
                        if not self.is_valid_mask(mask_tile):
                            continue
                        
                        # 创建文件路径
                        base_name = img_path.stem
                        row_off, col_off = window.row_off, window.col_off
                        
                        img_filename = Path(f"{self.output_dir}/images" + f"/{base_name}_crop_{row_off}_{col_off}.png")
                        mask_filename = Path(f"{self.output_dir}/masks" + f"/{base_name}_crop_{row_off}_{col_off}.png")
                        
                        # 如果文件不存在或需要重新保存，则创建
                        if self.save_cropped and (not img_filename.exists() or not mask_filename.exists()):
                            # 读取图像块
                            img_tile = img.read(window=window)
                            img_tile = np.transpose(img_tile, (1, 2, 0))  # 转换为 (H, W, C)
                            
                            # 转换为PIL图像并保存
                            Image.fromarray(img_tile.astype(np.uint8)).save(img_filename)
                            Image.fromarray(mask_tile.astype(np.uint8)).save(mask_filename)
                        
                        # 添加到路径列表中
                        image_paths.append(img_filename)
                        mask_paths.append(mask_filename)
                        count_data += 1
                        
                        if count_data >= max_count:
                            print(f"达到最大数据量 {max_count}，停止裁剪")
                            return image_paths, mask_paths
                
            except Exception as e:
                print(f"处理文件错误: {e}")
                continue
                
        return image_paths, mask_paths
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """按需加载图像和掩码"""
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # 使用with语句确保文件正确关闭
        with Image.open(img_path) as img, Image.open(mask_path) as mask:
            # 转换为RGB/L模式确保一致性
            img = img.convert('RGB')
            mask = mask.convert('L')
            
            # 应用数据增强
            if self.augment:
                state = torch.get_rng_state()
                img = self.augmentations(img)
                torch.set_rng_state(state)
                mask = self.augmentations(mask)
            
            # 转换为张量
            image_tensor = transforms.ToTensor()(img)
            mask_array = np.array(mask, dtype=np.int64)
            
            # 处理掩码值（确保是0和1）
            unique_vals = np.unique(mask_array)
            if len(unique_vals) > 2:
                # 如果有多个值，二值化处理（大于0的都为1）
                mask_array = (mask_array > 0).astype(np.int64)
            elif len(unique_vals) == 2 and not (0 in unique_vals and 1 in unique_vals):
                # 如果只有两个值但不是0和1，映射到0和1
                mask_array = (mask_array == unique_vals[1]).astype(np.int64)
            
            mask_tensor = torch.from_numpy(mask_array)
            
            # 应用额外的变换
            if self.transform:
                image_tensor = self.transform(image_tensor)
                
            return image_tensor, mask_tensor