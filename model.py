'''
2020/10/29
모델 새로 학습 시킬 때 output folder 지우고 val_result 안에 있는 사진 파일 지우거나 백업
videorighter
'''

from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger

setup_logger()
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from PIL import ImageFile
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_facelip_dtcs(json_dir):
    json_file = json_dir
    with open(json_file) as f:
        imgs_anns = json.load(f)
    dataset_dicts = []
    for z in range(len(imgs_anns['images'])):
        record = {}
        record['file_name'] = imgs_anns['images'][z]['file_name']
        record['image_id'] = imgs_anns['images'][z]['id']
        record['width'] = imgs_anns['images'][z]['width']
        record['height'] = imgs_anns['images'][z]['height']
        anno_list = []
        for i in range(len(imgs_anns['annotations'])):
            anno = {}
            if imgs_anns['images'][z]['id'] == imgs_anns['annotations'][i]['image_id']:
                anno['bbox'] = imgs_anns['annotations'][i]['bbox'].copy()  # check
                anno['bbox_mode'] = BoxMode.XYWH_ABS
                anno['segmentation'] = []
                anno['category_id'] = imgs_anns['annotations'][i]['category_id']-1  # check
                anno_list.append(anno)
        record['annotations'] = anno_list
        dataset_dicts.append(record)
    return dataset_dicts
dataset_dicts_train = get_facelip_dtcs("/home/videorighter/detectron/FACELIP_DATA_train/annotations/output_train.json")
print(len(dataset_dicts_train))

start = time.time()
############################### get resister, metadata #################################
for d in ["train", "val"]:
    DatasetCatalog.register("facelip_" + d, lambda d=d: get_facelip_dtcs(
        "/home/videorighter/detectron/FACELIP_DATA_" + d + "/annotations/output_" + d + ".json"))
    MetadataCatalog.get("facelip_" + d).set(thing_classes=["lip", "face", "product"])
train_facelip_metadata = MetadataCatalog.get("facelip_train")
val_facelip_metadata = MetadataCatalog.get("facelip_val")


################################# training model #######################################
cfg = get_cfg()
cfg.merge_from_file(
    "/home/videorighter/detectron/detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("facelip_train",)
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.MASK_ON = False
# cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.MODEL.WEIGHTS = "/home/videorighter/detectron/detectron2/configs/COCO-Detection/model_final_68b088.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.0001
cfg.SOLVER.MAX_ITER = 4000  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 2 classes (lip, face, product)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)

# 이전에 학습시킨 pth파일로 resume할 것인지 여부
trainer.resume_or_load(resume=True)
trainer.train()


################################# model test ####################################
cfg.DATASETS.TEST = ("facelip_val",)  # no metrics implemented for this dataset
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.2  # set the testing threshold for this model
predictor = DefaultPredictor(cfg)


############################# validation print ##################################
dataset_dicts_val = get_facelip_dtcs("/home/videorighter/detectron/FACELIP_DATA_val/annotations/output_val.json")
for d in dataset_dicts_val:
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=val_facelip_metadata,
                   scale=1)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = v.get_image()[:, :, ::-1]
    cv2.imwrite(os.path.join("/home/videorighter/detectron/val_result", os.path.split(d["file_name"])[1]), img)

############################# validation score ###################################
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

evaluator = COCOEvaluator("facelip_val", cfg, False, output_dir="./output_val/")
val_loader = build_detection_test_loader(cfg, "facelip_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
print("running time: ", time.time() - start)