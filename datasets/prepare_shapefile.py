import os
import rasterio
from rasterio import features
import geopandas as gpd
import numpy as np

def shapefile_to_mask(image_path, shapefile_path, out_mask_path, class_value=1, background=0):
    # 读取影像
    with rasterio.open(image_path) as src:
        meta = src.meta.copy()
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs

    # 读取shapefile并重投影到影像坐标系
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # rasterize
    mask = features.rasterize(
        [(geom, class_value) for geom in gdf.geometry],
        out_shape=(height, width),
        transform=transform,
        fill=background,
        dtype='uint8'
    )

    # 保存mask
    meta.update({'count': 1, 'dtype': 'uint8'})
    with rasterio.open(out_mask_path, 'w', **meta) as dst:
        dst.write(mask, 1)
    print(f"Mask saved to {out_mask_path}")

if __name__ == "__main__":
    # 修改为你的实际路径
    image_path = r"D:\your_data\your_image.tif"
    shapefile_path = r"D:\your_data\your_shapefile.shp"
    out_mask_path = r"D:\your_data\your_mask.tif"
    shapefile_to_mask(image_path, shapefile_path, out_mask_path, class_value=1, background=0)