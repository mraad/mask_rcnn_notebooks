#
# docker build -t mask_rcnn/mask_rcnn .
#
# FROM tensorflow/tensorflow:1.12.0-devel-gpu-py3
FROM tensorflow/tensorflow:1.12.0-devel-py3

RUN apt-get update && apt-get -y install vim wget python-pil python-lxml python-tk libsm6 libxrender-dev

RUN pip install --upgrade pip

RUN git clone --depth 1 https://github.com/matterport/Mask_RCNN.git /Mask_RCNN
RUN cd /Mask_RCNN &&\
 sed -i 's/^tensorflow.*/tensorflow==1.12.2/g' requirements.txt &&\
 pip install -r requirements.txt &&\
 pip install pycocotools

RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

EXPOSE 8888
EXPOSE 6006
