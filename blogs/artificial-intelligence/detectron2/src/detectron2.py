import detectron2

# import some common libraries
import numpy as np
import matplotlib.pyplot as plt
# import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


def inference(img_path='./images/team.jpg', ymlfile='COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml', panoptic=True):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(ymlfile))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(ymlfile)
    predictor = DefaultPredictor(cfg)
    im=plt.imread(img_path)
    outputs = predictor(im)
​
    # We can use `Visualizer` to draw the predictions on the image.
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    if panoptic:
        out = v.draw_panoptic_seg(outputs["panoptic_seg"][0].to('cpu'),outputs["panoptic_seg"][1])
        print(len(outputs["panoptic_seg"][1]))
    else:
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        print(len(outputs["instances"]))
    plt.figure()
    plt.title(ymlfile.split('/')[-1].split('.yml')[0])
    plt.imshow(out.get_image())
​

inference()
inference(ymlfile='COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml')
inference(ymlfile='COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml', panoptic=False)
inference(ymlfile='COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml', panoptic=False)
inference(ymlfile='COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml', panoptic=False)
