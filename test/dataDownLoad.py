# -*- coding: utf-8 -*-
import os
import math
import random
import io
import time
import numpy as np
from PIL import Image
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.mask import mask
from rasterio.windows import Window
import urllib.request
import concurrent.futures

# ================= 参数设置 =================
zoom = 18
shp_root = r"D:\crop_extraction_pipeline\data\边界目录"
save_root = r"D:\crop_extraction_pipeline\data\瓦片下载"
tile_size = 256
block_size = 50   # 每个下载块瓦片数（列/行）
retries = 3       # 单瓦片下载重试次数
pause_file = os.path.join(save_root, ".pause")  # 暂停标记文件

agents = [
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) Chrome/60.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) Chrome/12.0 Safari/534.27',
    'Mozilla/5.0 (Windows NT 5.1) Chrome/7.0 Safari/534.7'
]

# ================= 经纬度 <-> 瓦片转换 =================
def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile

def num2deg(x, y, zoom):
    n = 2.0 ** zoom
    lon_left = x / n * 360.0 - 180.0
    lon_right = (x+1) / n * 360.0 - 180.0
    lat_top = math.degrees(math.atan(math.sinh(math.pi * (1 - 2*y / n))))
    lat_bottom = math.degrees(math.atan(math.sinh(math.pi * (1 - 2*(y+1) / n))))
    return lon_left, lat_bottom, lon_right, lat_top

# ================= 下载单瓦片 =================
def download_tile(x, y, z, retries=retries):
    for i in range(retries):
        try:
            url = f"http://t0.tianditu.gov.cn/DataServer?T=img_w&x={x}&y={y}&l={z}&tk=5d22d49fdc586cb5caed68bfb12d1e6b"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', random.choice(agents))
            with urllib.request.urlopen(req, timeout=60) as resp:
                img_data = resp.read()
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            return img
        except Exception:
            print(f"瓦片下载失败 {x}_{y} 重试中 ({i+1}/{retries}) ...")
            time.sleep(1)
    return Image.new("RGB", (tile_size, tile_size), (255, 255, 255))

def download_tile_bytes(x, y, z, retries=retries):
    for i in range(retries):
        try:
            url = f"http://t0.tianditu.gov.cn/DataServer?T=img_w&x={x}&y={y}&l={z}&tk=5d22d49fdc586cb5caed68bfb12d1e6b"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', random.choice(agents))
            with urllib.request.urlopen(req, timeout=60) as resp:
                img_data = resp.read()
            return img_data
        except Exception:
            print(f"瓦片下载失败 {x}_{y} 重试中 ({i+1}/{retries}) ...")
            time.sleep(1)
    return None

# ================= 下载单块瓦片（顺序） =================
def download_block(dst, block_x_range, block_y_range, lefttop):
    total_tiles = len(block_x_range) * len(block_y_range)
    downloaded = 0
    tile_tasks = []
    tile_coords = []
    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for x in block_x_range:
            for y in block_y_range:
                if os.path.exists(pause_file):
                    print("检测到暂停文件，暂停下载...")
                    return False  # 停止当前下载块
                tile_coords.append((x, y))
                tile_tasks.append(executor.submit(download_tile_bytes, x, y, zoom))

        for idx, future in enumerate(concurrent.futures.as_completed(tile_tasks)):
            x, y = tile_coords[idx]
            img_data = future.result()
            if img_data is None:
                img = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
            else:
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
            arr = np.array(img)
            row_offset = (y - lefttop[1]) * tile_size
            col_offset = (x - lefttop[0]) * tile_size
            dst.write(arr[:,:,0], 1, window=Window(col_offset, row_offset, tile_size, tile_size))
            dst.write(arr[:,:,1], 2, window=Window(col_offset, row_offset, tile_size, tile_size))
            dst.write(arr[:,:,2], 3, window=Window(col_offset, row_offset, tile_size, tile_size))
            downloaded += 1
            percent = downloaded / total_tiles * 100
            print(f"{x}_{y} 下载完成 ({percent:.1f}%)")
    return True

# ================= 处理单个镇 =================
def process_town(shp_path):
    try:
        town_name = os.path.splitext(os.path.basename(shp_path))[0]
        town_dir = os.path.join(save_root, town_name)
        os.makedirs(town_dir, exist_ok=True)
        save_tif = os.path.join(town_dir, f"{town_name}_{zoom}.tif")
        save_clip_tif = os.path.join(town_dir, f"{town_name}_{zoom}_clip.tif")

        print(f"\n====== 处理 {town_name} ======")
        gdf = gpd.read_file(shp_path)
        minx, miny, maxx, maxy = gdf.total_bounds

        lefttop = deg2num(maxy, minx, zoom)
        rightbottom = deg2num(miny, maxx, zoom)
        cols = rightbottom[0] - lefttop[0] + 1
        rows = rightbottom[1] - lefttop[1] + 1
        print(f"瓦片范围: X {lefttop[0]}→{rightbottom[0]}, Y {lefttop[1]}→{rightbottom[1]} ({cols}x{rows})")

        min_lon, min_lat = num2deg(lefttop[0], rightbottom[1], zoom)[0:2]
        max_lon, max_lat = num2deg(rightbottom[0], lefttop[1], zoom)[2:4]
        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, cols*tile_size, rows*tile_size)

        with rasterio.open(
            save_tif, 'w',
            driver='GTiff',
            height=rows*tile_size,
            width=cols*tile_size,
            count=3,
            dtype='uint8',
            crs='EPSG:4326',
            transform=transform
        ) as dst:

            x_blocks = [range(x, min(x+block_size, rightbottom[0]+1)) for x in range(lefttop[0], rightbottom[0]+1, block_size)]
            y_blocks = [range(y, min(y+block_size, rightbottom[1]+1)) for y in range(lefttop[1], rightbottom[1]+1, block_size)]

            for bx in x_blocks:
                for by in y_blocks:
                    done_flag = os.path.join(town_dir, f"block_{bx.start}_{by.start}.done")
                    if os.path.exists(done_flag):
                        print(f"跳过已完成块 {bx.start}_{by.start}")
                        continue
                    success = download_block(dst, bx, by, lefttop)
                    if not success:
                        print(f"下载暂停于块 {bx.start}_{by.start}")
                        return
                    open(done_flag, 'w').close()

        print("GeoTIFF 下载完成:", save_tif)

        # 裁剪到矢量
        with rasterio.open(save_tif) as src:
            geoms = [geom for geom in gdf.geometry]
            out_image, out_transform = mask(src, geoms, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            with rasterio.open(save_clip_tif, "w", **out_meta) as dest:
                dest.write(out_image)

        print("裁剪完成:", save_clip_tif)

    except Exception as e:
        print(f"处理 {shp_path} 出错: {e}")

# ================= 遍历所有镇 =================
for root, dirs, files in os.walk(shp_root):
    for file in files:
        if file.endswith(".shp"):
            process_town(os.path.join(root, file))

print("\n全部完成 ✅")
