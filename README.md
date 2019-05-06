# Mask_RCNN Notebooks (WORK IN PROGRESS)

```bash
conda remove --yes --quiet --all --name mask_rcnn
conda create --yes --quiet --name mask_rcnn # --clone arcgispro-py3
conda activate mask_rcnn
conda install --yes --quiet pip python=3.6
# python -m pip install --upgrade pip
cd ${MRCNN_HOME}
sed -i '' 's/^tensorflow.*/tensorflow==1.12.2/g' requirements.txt
pip install -r requirements.txt
pip install pycocotools
```

```bash
docker run --runtime=nvidia --rm nvidia/cuda:9.2-runtime-ubuntu16.04 nvidia-smi
```

```bash
docker run --runtime=nvidia -it --rm tensorflow/tensorflow:1.12.0-devel-gpu-py3 \
   python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"
```
