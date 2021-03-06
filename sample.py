from keras.backend import clear_session
from mrcnn.config import Config
from mrcnn.model import MaskRCNN


class SampleConfig(Config):
    # Give the configuration a recognizable name
    NAME = "sample"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 class

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 50

    # Use a small epoch since the data is simple
    # STEPS_PER_EPOCH = 500

    # use small validation steps since the epoch is small
    # VALIDATION_STEPS = 5


global model

clear_session()  # Fix 1
config = SampleConfig()
model = MaskRCNN('inference', config, '.')  # Fix 2
