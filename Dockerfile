#
# docker build -t mask_rcnn/mask_rcnn .
#
FROM tensorflow/tensorflow:1.12.0-devel-gpu-py3
# FROM tensorflow/tensorflow:1.12.0-devel-py3

RUN apt-get update && apt-get -y install vim wget python-pil python-lxml python-tk libsm6 libxrender-dev

RUN pip install --upgrade pip

RUN git clone --depth 1 https://github.com/matterport/Mask_RCNN.git /Mask_RCNN
RUN cd /Mask_RCNN &&\
 sed -i 's/tensorflow>=1.3.0/tensorflow=gpu==1.8/g' requirements.txt &&\
 pip install -r requirements.txt &&\
 pip install pycocotools

ENV MRCNN_HOME=/Mask_RCNN
COPY tf_keras_gpu_version.ipynb /Mask_RCNN
COPY train2.ipynb /Mask_RCNN

RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

EXPOSE 8888
EXPOSE 6006
