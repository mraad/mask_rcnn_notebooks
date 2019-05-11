import glob
import os
import sys
from functools import lru_cache

import numpy as np
import skimage

MRCNN_DIR = os.getenv("MRCNN_HOME", "Mask_RCNN")
sys.path.append(MRCNN_DIR)

from mrcnn.config import Config
from mrcnn import utils

COCO_MODEL_PATH = os.path.join(MRCNN_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

IMG_SIZE = 256
MODEL_DIR = "logs"


class TrainConfig(Config):
    NAME = "mr"
    BATCH_SIZE = 8
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    IMAGE_MIN_DIM = IMG_SIZE
    IMAGE_MAX_DIM = IMG_SIZE
    NUM_CLASSES = 1 + 1  # Background + 1 class
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    RPN_ANCHOR_SCALES = (10, 20, 40, 80, 160)
    STEPS_PER_EPOCH = 50
    VALIDATION_STEPS = 10
    TRAIN_ROIS_PER_IMAGE = 200
    DETECTION_MAX_INSTANCES = 30
    MAX_GT_INSTANCES = 30
    # MEAN_PIXEL = np.array([150.1, 143.6, 130.3])
    MEAN_PIXEL = np.array([130.2, 126.0, 123.8])
    LEARNING_RATE = 1.0e-4
    WEIGHT_DECAY = 1.0e-5


class InferenceConfig(TrainConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.9
    DETECTION_NMS_THRESHOLD = 0.2


class MRDataset(utils.Dataset):
    def load_glob(self, tif_glob):

        self.add_class("mr", 1, "E_PipeCompleted")

        for image_id, tif_path in enumerate(tif_glob):
            _, tif_name = os.path.split(tif_path)
            base, name = os.path.split(tif_path)
            base, _ = os.path.split(base)
            mask = os.path.join(base, "labels", "E_PipeCompleted", name)

            self.add_image("mr",
                           image_id=image_id,
                           path=tif_path,
                           mask=mask,
                           width=IMG_SIZE,
                           height=IMG_SIZE,
                           filename=tif_name)

    def load(self, base_path):
        tif_glob = glob.glob(os.path.join(base_path, "*", "images", "*.tif"))
        self.load_glob(tif_glob)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "mr":
            return info["path"]
        else:
            super(self.__class__).image_reference(self, image_id)

    @lru_cache(maxsize=None)
    def imread(self, image_path):
        if os.path.exists(image_path):
            image = skimage.io.imread(image_path)
            # Sometimes the depth is larger than 1 from ETD4DL
            if len(image.shape) > 2:
                image = image[:,:,0]
            return image
        else:
            return np.zeros([IMG_SIZE, IMG_SIZE, 1], dtype=np.int8)

    # def load_mask(self, image_id):
    #     info = self.image_info[image_id]
    #     tif_path = info["path"]
    #     masks = []
    #     clazz = []
    #     for class_info in self.class_info:
    #         class_nm = class_info["name"]
    #         class_id = class_info["id"]
    #         base, name = os.path.split(tif_path)
    #         base, _ = os.path.split(base)
    #         name = os.path.join(base, "labels", class_nm, name)
    #         if os.path.exists(name):
    #             mask = self.imread(name)
    #             instance_ids = np.unique(mask)
    #             for i in instance_ids:
    #                 if i > 0:
    #                     m = np.zeros(mask.shape)
    #                     m[mask == i] = i
    #                     if np.any(m == i):
    #                         masks.append(m)
    #                         clazz.append(class_id)
    #     if masks:
    #         masks = np.stack(masks, axis=-1)
    #     else:
    #         masks = np.array(masks)
    #     return masks.astype(np.bool), np.array(clazz, dtype=np.int32)

    def load_mask_image(self, image_id):
        info = self.image_info[image_id]
        mask = info["mask"] 
        return self.imread(mask)
            
    @lru_cache(maxsize=None)
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask = info["mask"]
        image = self.imread(mask)
        label = np.unique(image)
        count = label.size - 1
        if count > 0:
            masks = np.zeros([IMG_SIZE, IMG_SIZE, count], dtype=np.bool)
            clazz = np.ones(count, np.int32)
            for i in range(1, label.size):
                idx = image == i
                idx = idx.reshape(image.shape)
                masks[:, :, i - 1] = idx
        else:
            masks = np.zeros([IMG_SIZE, IMG_SIZE, 1], dtype=np.bool)
            clazz = np.ones(1, np.int32)
        return masks, clazz
