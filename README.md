# Mask_RCNN Notebooks

Project to perform object detection and segmentation within ArcGIS using [Mask RCNN](https://github.com/matterport/Mask_RCNN).

### Setup Mask RCNN

Clone the [Mask RCNN](https://github.com/matterport/Mask_RCNN) project locally, and define an environment variable named `MRCNN_HOME` to the cloned location.

```bash
git clone https://github.com/matterport/Mask_RCNN
cd Mask_RCNN
export MRCNN_HOME=$PWD
```

### Setup ArcGIS Pro Env

Start a `Python Command Prompt` From the ArcGIS Start menu item, and execute the following commands:

```bash
conda remove --yes --quiet --all --name mask_rcnn
conda create --yes --quiet --name mask_rcnn --clone arcgispro-py3
conda activate mask_rcnn
conda install --yes --quiet pip python=3.6
```

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
python -m pip install --upgrade pip --user
cd ${MRCNN_HOME}
sed -i '' 's/^tensorflow.*/tensorflow==1.12.2/g' requirements.txt
sed -i '' 's/^tensorflow.*/#/g' requirements.txt
pip install -r requirements.txt
pip install cachetools
pip install pycocotools
```

Make sure to run the `shapes` notebook in the `${MRCNN_HOME}\samples` folder to test the installation.

### Misc

```bash
docker run --runtime=nvidia --rm nvidia/cuda:9.2-runtime-ubuntu16.04 nvidia-smi
```

```bash
docker run --runtime=nvidia -it --rm tensorflow/tensorflow:1.12.0-devel-gpu-py3 \
   python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"
```
