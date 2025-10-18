from PIL import Image
import numpy as np

# 加载mask图像
mask_path = r"F:\yanRuyou\cropped\masks\白岭镇_18_crop_0_12288.png"
mask = Image.open(mask_path)
mask_array = np.array(mask)

# 打印像素值统计
print("像素值统计:")
unique, counts = np.unique(mask_array, return_counts=True)
for val, count in zip(unique, counts):
    print(f"值 {val}: {count} 像素 ({count/mask_array.size*100:.2f}%)")

# 可视化mask
import matplotlib.pyplot as plt

plt.imshow(mask_array, cmap='gray')
plt.title("Mask Visualization")
plt.colorbar()
plt.show()