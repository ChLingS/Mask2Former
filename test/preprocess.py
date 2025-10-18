import os
from typing import List
from tqdm import tqdm

def create_buffer(shp_path: str, buffer_distance: float = 50.0) -> None:
    pass

def search_dependency_shp(area_list: List[str], search_dir: str) -> List[str]:
    """搜索匹配的Shapefile"""
    all_shps = []
    
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.endswith('.shp'):
                file_name = os.path.splitext(file)[0]
                for area in area_list:
                    if area in file_name:
                        all_shps.append(os.path.join(root, file))
                        break
    
    return all_shps

if __name__ == "__main__":
    processArea = ['上衫乡', '南坑镇', '古市镇', '四都镇', 
                   '天九镇', '太阳升镇', '宁州镇', '庙岭乡', 
                   '排上镇', '新泉乡', '新湾乡', '杭口镇', 
                   '桐坪镇', '江口镇', '浮槎乡', '湘东镇', '版石镇', 
                   '白岭镇', '白鹭乡', '石坳乡', '赤山镇', '路口乡']
    
    shp_box = search_dependency_shp(processArea, r'E:\帮人做东西\严茹尤\训练数据\训练数据')
    
    print(f"找到 {len(shp_box)} 个匹配的Shapefile")
    
    for shp_path in tqdm(shp_box, desc="创建50米内缓冲区"):
        try:
            pass
        except Exception as e:
            print(f"处理文件 {shp_path} 时出错: {e}")
            continue
    
    print("处理完成！")