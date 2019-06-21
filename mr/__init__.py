import glob
import os
import sys
from functools import lru_cache

import numpy as np
import skimage

# from cachetools import cached

MRCNN_DIR = os.getenv("MRCNN_HOME", "Mask_RCNN")
sys.path.append(MRCNN_DIR)

from mrcnn.config import Config
from mrcnn import utils

COCO_MODEL_PATH = os.path.join(MRCNN_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

MODEL_DIR = "logs"
IMG_SIZE = 512


class TrainConfig(Config):
    NAME = "mr"
    BACKBONE = "resnet50"
    BATCH_SIZE = 8
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4
    IMAGE_MIN_DIM = IMG_SIZE
    IMAGE_MAX_DIM = IMG_SIZE
    NUM_CLASSES = 1 + 1  # Background + 1 class
    RPN_ANCHOR_RATIOS = [0.1, 0.25, 1, 4, 10]
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    # RPN_ANCHOR_SCALES = (10, 20, 40, 80, 160)
    STEPS_PER_EPOCH = 50
    VALIDATION_STEPS = 10
    TRAIN_ROIS_PER_IMAGE = 200
    DETECTION_MAX_INSTANCES = 30
    MAX_GT_INSTANCES = 30
    # MEAN_PIXEL = np.array([150.1, 143.6, 130.3]) # coco
    # MEAN_PIXEL = np.array([130.2, 126.0, 123.8]) #Tanks 256
    MEAN_PIXEL = np.array([122.4, 119.5, 118.1])  # Pipes 512
    LEARNING_RATE = 1.0e-4
    WEIGHT_DECAY = 1.0e-5
    # USE_MINI_MASK = False
    LOSS_WEIGHTS = {'rpn_class_loss': 1.0, 'rpn_bbox_loss': 1.0, 'mrcnn_class_loss': 1.0, 'mrcnn_bbox_loss': 1.0,
                    'mrcnn_mask_loss': 10.0}


class InferenceConfig(TrainConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # DETECTION_MIN_CONFIDENCE = 0.9
    # DETECTION_NMS_THRESHOLD = 0.2


class MRDataset(utils.Dataset):
    def load_glob(self, tif_glob):

        self.add_class("mr", 1, "Tank")

        for image_id, tif_path in enumerate(tif_glob):
            _, tif_name = os.path.split(tif_path)
            base, name = os.path.split(tif_path)
            base, _ = os.path.split(base)
            mask = os.path.join(base, "labels", "Tank", name)

            self.add_image("mr",
                           image_id=image_id,
                           path=tif_path,
                           mask=mask,
                           width=IMG_SIZE,
                           height=IMG_SIZE,
                           filename=tif_name)

    def load(self, base_path):
        tif_glob = glob.glob(os.path.join(base_path, "images", "*.png"))
        self.load_glob(tif_glob)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "mr":
            return info["path"]
        else:
            super(self.__class__).image_reference(self, image_id)

    @lru_cache(maxsize=None)
    # @cached(cache={})
    def imread(self, image_path):
        if os.path.exists(image_path):
            image = skimage.io.imread(image_path)
            # Sometimes the depth is larger than 1 from ETD4DL
            if image.ndim > 2:
                image = image[:, :, 0]
            return image
        else:
            return np.zeros([IMG_SIZE, IMG_SIZE, 1], dtype=np.int8)

    @lru_cache(maxsize=None)
    # @cached(cache={})
    def load_mask_image(self, image_id):
        info = self.image_info[image_id]
        mask = info["mask"]
        return self.imread(mask)

    @lru_cache(maxsize=None)
    # @cached(cache={})
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
