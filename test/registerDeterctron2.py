import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg

# 关键：导入Mask2Former的Trainer
from mask2former import MaskFormer

def get_semantic_dicts(img_dir, mask_dir):
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.tif')])
    dataset_dicts = []
    for img_file in img_files:
        record = {
            "file_name": os.path.join(img_dir, img_file),
            "sem_seg_file_name": os.path.join(mask_dir, img_file),
            "image_id": img_file.split('.')[0]
        }
        dataset_dicts.append(record)
    return dataset_dicts

if __name__ == "__main__":
    # 数据集路径
    img_dir = r"F:\mask2formerMiddata\images"
    mask_dir = r"F:\mask2formerMiddata\masks"
    DatasetCatalog.register("my_semseg_train", lambda: get_semantic_dicts(img_dir, mask_dir))
    MetadataCatalog.get("my_semseg_train").set(
        stuff_classes=["background", "field"],
        evaluator_type="sem_seg",
        ignore_label=0
    )
    print("Detectron2数据集注册完成：my_semseg_train")

    # 配置Mask2Former
    cfg = get_cfg()
    cfg.merge_from_file(r"d:\Mask2Former\configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml")
    cfg.DATASETS.TRAIN = ("my_semseg_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2  # 类别数（背景+地块）
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 1000
    cfg.OUTPUT_DIR = "./output_maskformer"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 用Mask2Former的Trainer
    trainer = MaskFormer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
