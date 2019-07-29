# Mask_RCNN Notebooks

Project to perform object detection and segmentation within ArcGIS using [Mask RCNN](https://github.com/matterport/Mask_RCNN).

### Setup Mask RCNN

Clone the [Mask RCNN](https://github.com/matterport/Mask_RCNN) project locally, and define an environment variable named `MRCNN_HOME` to the cloned location.

```bash
git clone https://github.com/matterport/Mask_RCNN
cd Mask_RCNN
export MRCNN_HOME=$PWD
```

### Setup Standalone Env

```
conda config --set channel_priority false
conda create --yes --name mask_rcnn python=3.6
conda activate mask_rcnn
```

### Setup ArcGIS Pro Env

Start a `Python Command Prompt` From the ArcGIS Start menu item, and execute the following commands:

```bash
conda remove --yes --all --name mask_rcnn
conda create --yes --name mask_rcnn --clone arcgispro-py3
activate mask_rcnn
conda install --yes pip python=3.6
python -m pip install --upgrade pip --user
```

### Installing Tensorflow

Execute the following to install the CPU version of tensorflow:

```bash
conda install -c anaconda 'tensorflow=1.12*=mkl*'
```

Execute the following to install the GPU version of tensorflow:

```bash
conda install -c anaconda 'tensorflow=1.12*=gpu*'
```

Make sure to remove the `tensorflow` entry in `requirements.txt` file in the `Mask_RCNN` folder.

```
conda install -c conda-forge jupyterlab
cd ${MRCNN_HOME}
sed -i '' 's/^tensorflow.*/#/g' requirements.txt
pip install -r requirements.txt
# Optional
pip install cachetools
pip install pycocotools
```

Make sure to run the `shapes` notebook in the `${MRCNN_HOME}/samples` folder to test the installation.

The following is a sample EMD (Esri Model Definition) file to be used with ArcGIS Pro [Detect Objects Using Deep Learning](https://pro.arcgis.com/en/pro-app/tool-reference/image-analyst/detect-objects-using-deep-learning.htm):

```json
{
  "Framework": "Keras",
  "ModelFile": "mask_rcnn_sample_0085.h5",
  "ModelConfiguration": {
    "Name": "MaskRCNN",
    "Config": "sample",
    "Architecture": "sample"
  },
  "ModelType": "ObjectDetection",
  "ImageWidth": 256,
  "ImageHeight": 256,
  "ExtractBands": [0, 1, 2],
  "Classes": [
    {
      "Value": 1,
      "Name": "MyObject",
      "Color": [0, 255, 0]
    }
  ]
}
```

The following is a sample configuration (`sample.py`) associated with above EMD:

```python
from keras.backend import clear_session
from mrcnn.config import Config
from mrcnn.model import MaskRCNN


class SampleConfig(Config):
    NAME = "sample"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # background + 1 class
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256


global model

clear_session()
config = SampleConfig()
model = MaskRCNN('inference', config, '.')
```

### Notes To Self

```bash
docker run --runtime=nvidia --rm nvidia/cuda:9.2-runtime-ubuntu16.04 nvidia-smi
```

```bash
docker run --runtime=nvidia -it --rm tensorflow/tensorflow:1.12.0-devel-gpu-py3 \
   python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"
```

```python
# http://code.activestate.com/recipes/578231-probably-the-fastest-memoization-decorator-in-the-/
# def memoize(f):
#     """ Memoization decorator for a function taking a single argument """
#
#     class memoize(dict):
#         def __missing__(self, key):
#             ret = self[key] = f(key)
#             return ret
#
#     return memoize().__getitem__
```